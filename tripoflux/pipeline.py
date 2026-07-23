"""End-to-end pipeline: text prompt → image → background removal → 3D Gaussian Splat."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

from PIL import Image

from .models.birefnet_mlx import BiRefNetMLX
from .models.flux_klein_mlx import FluxGenerationConfig, FluxKleinGenerator
from .models.triposplat_wrapped import SplatGenerationConfig, TripoSplatGenerator

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, float], None]


@dataclass
class PipelineConfig:
    triposplat_dir: Union[str, Path]
    flux_backend: str = "mlx"
    birefnet_backend: str = "mlx"
    triposplat_backend: str = "mps"
    flux_quantize: int = 8
    image_width: int = 1024
    image_height: int = 1024
    flux_steps: int = 4
    flux_guidance: float = 1.0
    splat_steps: int = 20
    splat_guidance: float = 3.0
    splat_shift: float = 3.0
    num_gaussians: int = 262144
    seed: int = 42


@dataclass
class PipelineResult:
    prompt: str
    generated_image: Image.Image
    rgba_image: Image.Image
    prepared_image: Image.Image
    ply_bytes: bytes
    splat_bytes: bytes
    spz_bytes: bytes
    metadata: dict = field(default_factory=dict)


class TripoFluxPipeline:
    """Full text → image → RMBG → Gaussian Splat pipeline."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.flux = FluxKleinGenerator(
            backend=cfg.flux_backend,
            quantize=cfg.flux_quantize,
        )
        self.birefnet = BiRefNetMLX(
            triposplat_dir=cfg.triposplat_dir,
            backend=cfg.birefnet_backend,
        )
        self.triposplat = TripoSplatGenerator(
            ckpt_path=Path(cfg.triposplat_dir) / "diffusion_models" / "triposplat_fp16.safetensors",
            decoder_path=Path(cfg.triposplat_dir) / "vae" / "triposplat_vae_decoder_fp16.safetensors",
            dinov3_path=Path(cfg.triposplat_dir) / "clip_vision" / "dino_v3_vit_h.safetensors",
            flux2_vae_encoder_path=Path(cfg.triposplat_dir) / "vae" / "flux2-vae.safetensors",
            rmbg_path=Path(cfg.triposplat_dir) / "background_removal" / "birefnet.safetensors",
            device=cfg.triposplat_backend,
        )

    def _progress(self, cb: Optional[ProgressCallback], stage: str, frac: float) -> None:
        if cb is not None:
            try:
                cb(stage, max(0.0, min(1.0, frac)))
            except Exception:
                pass

    def generate_image(
        self,
        prompt: str,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        flux_quantize: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> Image.Image:
        self._progress(progress, "flux", 0.05)
        flux = self.flux
        # If the caller requests a different quantization level, build a
        # temporary generator for this call. Model weights are reloaded
        # by mflux, so this is expensive and should be used sparingly.
        if flux_quantize is not None and flux_quantize != self.cfg.flux_quantize:
            flux = FluxKleinGenerator(
                backend=self.cfg.flux_backend,
                quantize=flux_quantize,
            )
        img = flux.generate(
            FluxGenerationConfig(
                prompt=prompt,
                width=width or self.cfg.image_width,
                height=height or self.cfg.image_height,
                num_inference_steps=self.cfg.flux_steps,
                guidance_scale=self.cfg.flux_guidance,
                seed=seed if seed is not None else self.cfg.seed,
                negative_prompt=negative_prompt,
            )
        )
        self._progress(progress, "flux", 1.0)
        return img

    def remove_background(
        self,
        image: Image.Image,
        progress: Optional[ProgressCallback] = None,
    ) -> Image.Image:
        self._progress(progress, "birefnet", 0.05)
        rgba = self.birefnet.remove_background(image)
        self._progress(progress, "birefnet", 1.0)
        return rgba

    def generate_splat(
        self,
        rgba_image: Image.Image,
        num_gaussians: Optional[int] = None,
        seed: Optional[int] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> Tuple[bytes, bytes, bytes, Image.Image]:
        self._progress(progress, "triposplat", 0.05)
        ply, splat, spz, prepared = self.triposplat.image_to_splat(
            rgba_image,
            cfg=SplatGenerationConfig(
                num_gaussians=num_gaussians or self.cfg.num_gaussians,
                seed=seed if seed is not None else self.cfg.seed,
                steps=self.cfg.splat_steps,
                guidance_scale=self.cfg.splat_guidance,
                shift=self.cfg.splat_shift,
            ),
        )
        self._progress(progress, "triposplat", 1.0)
        return ply, splat, spz, prepared

    def run(
        self,
        prompt: str,
        seed: Optional[int] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        num_gaussians: Optional[int] = None,
        flux_quantize: Optional[int] = None,
        negative_prompt: Optional[str] = None,
        progress: Optional[ProgressCallback] = None,
    ) -> PipelineResult:
        """Run the full pipeline from prompt to Gaussian Splat."""
        self._progress(progress, "flux", 0.0)
        image = self.generate_image(
            prompt, seed=seed, width=width, height=height,
            flux_quantize=flux_quantize, negative_prompt=negative_prompt,
            progress=progress
        )

        self._progress(progress, "birefnet", 0.0)
        rgba = self.remove_background(image, progress=progress)

        self._progress(progress, "triposplat", 0.0)
        ply, splat, spz, prepared = self.generate_splat(
            rgba, num_gaussians=num_gaussians, seed=seed, progress=progress
        )

        return PipelineResult(
            prompt=prompt,
            generated_image=image,
            rgba_image=rgba,
            prepared_image=prepared,
            ply_bytes=ply,
            splat_bytes=splat,
            spz_bytes=spz,
            metadata={
                "seed": seed if seed is not None else self.cfg.seed,
                "width": width or self.cfg.image_width,
                "height": height or self.cfg.image_height,
                "num_gaussians": num_gaussians or self.cfg.num_gaussians,
                "flux_quantize": flux_quantize or self.cfg.flux_quantize,
                "negative_prompt": negative_prompt,
                "flux_backend": self.cfg.flux_backend,
                "birefnet_backend": self.cfg.birefnet_backend,
                "triposplat_backend": self.cfg.triposplat_backend,
            },
        )


def load_pipeline_from_yaml(config_path: Union[str, Path]) -> TripoFluxPipeline:
    import yaml

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    models = data.get("models", {})
    backend = data.get("backend", {})
    quantization = data.get("quantization", {})
    inference = data.get("inference", {})

    cfg = PipelineConfig(
        triposplat_dir=models.get("triposplat_dir", "ckpts/VAST-AI/TripoSplat"),
        flux_backend=backend.get("flux", "mlx"),
        birefnet_backend=backend.get("birefnet", "mlx"),
        triposplat_backend=backend.get("triposplat", "mps"),
        flux_quantize=quantization.get("flux_bits", 8),
        image_width=inference.get("image_width", 1024),
        image_height=inference.get("image_height", 1024),
        flux_steps=inference.get("num_inference_steps", 4),
        flux_guidance=inference.get("guidance_scale", 1.0),
        splat_steps=inference.get("triposplat_steps", 20),
        splat_guidance=inference.get("triposplat_guidance_scale", 3.0),
        splat_shift=inference.get("triposplat_shift", 3.0),
        num_gaussians=inference.get("num_gaussians", 262144),
        seed=inference.get("seed", 42),
    )
    return TripoFluxPipeline(cfg)
