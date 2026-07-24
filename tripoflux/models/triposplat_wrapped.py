"""TripoSplat wrapper for image → 3D Gaussian Splat conversion.

Loads the official TripoSplat checkpoints and exposes a clean API for
generating .ply / .splat outputs from a background-removed image.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
from PIL import Image

from ..vendor.triposplat.triposplat import TripoSplatPipeline
from .triposplat_mlx_pipeline import TripoSplatHybridPipeline

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplatGenerationConfig:
    num_gaussians: int = 262144
    seed: int = 42
    steps: int = 28
    guidance_scale: float = 3.0
    shift: float = 3.0
    erode_radius: int = 1


class TripoSplatGenerator:
    """Generate 3D Gaussian Splats from images using TripoSplat.

    Supports two backends:
    - ``mps``: run the original PyTorch implementation on Apple Metal.
    - ``mlx``: run the hybrid MLX/MPS pipeline (MLX encoders + MPS decoder).
    """

    def __init__(
        self,
        ckpt_path: Union[str, Path],
        decoder_path: Union[str, Path],
        dinov3_path: Union[str, Path],
        flux2_vae_encoder_path: Union[str, Path],
        rmbg_path: Union[str, Path],
        device: str = "mps",
        dtype: torch.dtype = torch.float16,
    ):
        self.device = torch.device(device) if device != "mlx" else None
        self.dtype = dtype
        self.backend = device
        if device == "mlx":
            self._pipeline = TripoSplatHybridPipeline(
                ckpt_path=ckpt_path,
                decoder_path=decoder_path,
                dinov3_path=dinov3_path,
                flux2_vae_encoder_path=flux2_vae_encoder_path,
                rmbg_path=rmbg_path,
                device="mps",
                use_mlx_encoders=True,
            )
        else:
            self._pipeline = TripoSplatPipeline(
                ckpt_path=str(ckpt_path),
                decoder_path=str(decoder_path),
                dinov3_path=str(dinov3_path),
                flux2_vae_encoder_path=str(flux2_vae_encoder_path),
                rmbg_path=str(rmbg_path),
                device=device,
            )

    def preprocess_image(self, image: Image.Image, erode_radius: int = 1) -> Image.Image:
        return self._pipeline.preprocess_image(image, erode_radius=erode_radius)

    def image_to_splat(
        self,
        image: Image.Image,
        cfg: SplatGenerationConfig = None,
        show_progress: bool = False,
        callback=None,
        preview_callback=None,
    ) -> Tuple[bytes, bytes, bytes, Image.Image]:
        """Convert an RGBA image to Gaussian Splat bytes.

        Args:
            callback: Optional ``fn(step, total)`` invoked after each sampler step.
            preview_callback: Optional ``fn(step, splat_bytes)`` invoked with a
                lightweight intermediate splat every few sampler steps
                (MLX/hybrid backend only).

        Returns:
            (ply_bytes, splat_bytes, spz_bytes, prepared_image)
        """
        if cfg is None:
            cfg = SplatGenerationConfig()

        if self.backend == "mlx":
            ply_bytes, splat_bytes, spz_bytes, prepared = self._pipeline.image_to_splat(
                image,
                cfg=cfg,
                show_progress=show_progress,
                callback=callback,
                preview_callback=preview_callback,
            )
        else:
            gaussian, prepared = self._pipeline.run(
                image,
                seed=cfg.seed,
                steps=cfg.steps,
                guidance_scale=cfg.guidance_scale,
                shift=cfg.shift,
                num_gaussians=cfg.num_gaussians,
                erode_radius=cfg.erode_radius,
                show_progress=show_progress,
                callback=callback,
            )
            # SPZ is the canonical artifact; ply/splat are derived from it
            # (see the mlx path in TripoSplatHybridPipeline).
            from .spz_utils import (
                gaussian_to_spz_bytes,
                spz_bytes_to_ply_bytes,
                spz_bytes_to_splat_bytes,
            )
            spz_bytes = gaussian_to_spz_bytes(gaussian)
            ply_bytes = spz_bytes_to_ply_bytes(spz_bytes)
            splat_bytes = spz_bytes_to_splat_bytes(spz_bytes)
        return ply_bytes, splat_bytes, spz_bytes, prepared

    def save_splat(
        self,
        image: Image.Image,
        output_path: Union[str, Path],
        cfg: SplatGenerationConfig = None,
    ) -> Path:
        ply_bytes, splat_bytes, spz_bytes, _ = self.image_to_splat(image, cfg=cfg)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if output_path.suffix.lower() == ".ply":
            output_path.write_bytes(ply_bytes)
        elif output_path.suffix.lower() == ".spz":
            output_path.write_bytes(spz_bytes)
        else:
            output_path.write_bytes(splat_bytes)
        return output_path


def create_triposplat_generator(
    triposplat_dir: Union[str, Path],
    device: str = "mps",
) -> TripoSplatGenerator:
    """Create a TripoSplatGenerator from the downloaded checkpoint directory.

    The directory should contain:
        - diffusion_models/triposplat_fp16.safetensors
        - clip_vision/dino_v3_vit_h.safetensors
        - vae/triposplat_vae_decoder_fp16.safetensors
        - vae/flux2-vae.safetensors
        - background_removal/birefnet.safetensors

    Args:
        device: ``"mps"`` for the original PyTorch backend, or ``"mlx"``
            for the hybrid MLX/MPS backend with MLX-accelerated encoders.
    """
    triposplat_dir = Path(triposplat_dir)
    return TripoSplatGenerator(
        ckpt_path=triposplat_dir / "diffusion_models" / "triposplat_fp16.safetensors",
        decoder_path=triposplat_dir / "vae" / "triposplat_vae_decoder_fp16.safetensors",
        dinov3_path=triposplat_dir / "clip_vision" / "dino_v3_vit_h.safetensors",
        flux2_vae_encoder_path=triposplat_dir / "vae" / "flux2-vae.safetensors",
        rmbg_path=triposplat_dir / "background_removal" / "birefnet.safetensors",
        device=device,
    )
