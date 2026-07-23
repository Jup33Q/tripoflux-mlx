"""Hybrid MLX/MPS TripoSplat pipeline.

Uses MLX for the ported components (DinoV3ViT, Flux2VAEEncoder) and
PyTorch MPS for the remaining components (flow model, decoder).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Union

import mlx.core as mx
import numpy as np
import torch
from PIL import Image

from ..vendor.triposplat.triposplat import TripoSplatPipeline
from .dinov3_mlx import DinoV3ViT as DinoV3ViTMLX
from .flux2vae_mlx import Flux2VAEEncoder as Flux2VAEEncoderMLX
from .flow_mlx import LatentSeqMMFlowModel as LatentSeqMMFlowModelMLX

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HybridSplatConfig:
    num_gaussians: int = 262144
    seed: int = 42
    steps: int = 20
    guidance_scale: float = 3.0
    shift: float = 3.0
    erode_radius: int = 1


class TripoSplatHybridPipeline:
    """TripoSplat pipeline with MLX-accelerated encoders and MPS decoder."""

    def __init__(
        self,
        ckpt_path: Union[str, Path],
        decoder_path: Union[str, Path],
        dinov3_path: Union[str, Path],
        flux2_vae_encoder_path: Union[str, Path],
        rmbg_path: Union[str, Path],
        device: str = "mps",
        use_mlx_encoders: bool = True,
    ):
        self.device = torch.device(device)
        self.use_mlx_encoders = use_mlx_encoders

        # Load the full PyTorch pipeline for components not yet ported.
        self._torch_pipeline = TripoSplatPipeline(
            ckpt_path=str(ckpt_path),
            decoder_path=str(decoder_path),
            dinov3_path=str(dinov3_path),
            flux2_vae_encoder_path=str(flux2_vae_encoder_path),
            rmbg_path=str(rmbg_path),
            device=device,
        )

        # Optionally replace encoders with MLX versions.
        self._dinov3_mlx: Optional[DinoV3ViTMLX] = None
        self._vae_mlx: Optional[Flux2VAEEncoderMLX] = None
        self._flow_mlx: Optional[LatentSeqMMFlowModelMLX] = None
        if use_mlx_encoders:
            try:
                self._dinov3_mlx = DinoV3ViTMLX()
                self._dinov3_mlx.load_safetensors(str(dinov3_path))
                self._vae_mlx = Flux2VAEEncoderMLX()
                self._vae_mlx.load_safetensors(str(flux2_vae_encoder_path))
                logger.info("Loaded MLX DinoV3ViT and Flux2VAEEncoder")
            except Exception as exc:
                logger.warning("Failed to load MLX encoders, falling back to MPS: %s", exc)
                self._dinov3_mlx = None
                self._vae_mlx = None

            try:
                self._flow_mlx = LatentSeqMMFlowModelMLX(
                    q_token_length=8192, in_channels=16, cam_channels=5, out_channels=16,
                    model_channels=1024, cond_channels=1280, cond2_channels=128,
                    num_refiner_blocks=2, num_blocks=24, num_heads=16, mlp_ratio=4,
                    qk_rms_norm=True, share_mod=True, use_shift_table=True,
                )
                self._flow_mlx.load_safetensors(str(ckpt_path))
                logger.info("Loaded MLX LatentSeqMMFlowModel")
            except Exception as exc:
                logger.warning("Failed to load MLX flow model, falling back to MPS: %s", exc)
                self._flow_mlx = None

    def preprocess_image(self, image: Image.Image, erode_radius: int = 1) -> Image.Image:
        return self._torch_pipeline.preprocess_image(image, erode_radius=erode_radius)

    def encode_image_mlx(self, image: Image.Image) -> dict:
        """Encode image using MLX encoders, returning MLX arrays."""
        if self._dinov3_mlx is None or self._vae_mlx is None:
            raise RuntimeError("MLX encoders not loaded")

        import torchvision.transforms as T

        img_tensor = T.ToTensor()(image).unsqueeze(0)
        img_normed = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img_tensor)
        img_mlx = mx.array(img_normed.numpy())

        dinov3_feat = self._dinov3_mlx(img_mlx)
        # Layer norm over last dim
        mean = mx.mean(dinov3_feat, axis=-1, keepdims=True)
        var = mx.var(dinov3_feat, axis=-1, keepdims=True)
        dinov3_feat = (dinov3_feat - mean) / mx.sqrt(var + 1e-5)

        vae_input = mx.array((img_tensor.numpy() * 2 - 1))
        vae_feat = self._vae_mlx.encode(vae_input, deterministic=False, seed=None)

        # Pad 5 zero tokens to match feature1 length
        zero_reg = mx.zeros((vae_feat.shape[0], 5, vae_feat.shape[2]), dtype=vae_feat.dtype)
        vae_feat = mx.concatenate([zero_reg, vae_feat], axis=1)

        return {"feature1": dinov3_feat, "feature2": vae_feat}

    def _sample_latent_mlx(
        self,
        cond: dict,
        steps: int = 20,
        guidance_scale: float = 3.0,
        shift: float = 3.0,
        seed: int = 42,
        show_progress: bool = False,
        callback=None,
    ) -> dict:
        """Run the Euler CFG sampler using the MLX flow model."""
        if self._flow_mlx is None:
            raise RuntimeError("MLX flow model not loaded")

        mx.random.seed(seed)
        noise = {
            'latent': mx.random.normal((1, 8192, 16)),
            'camera': mx.random.normal((1, 1, 5)),
        }
        neg_cond = {k: mx.zeros_like(v) for k, v in cond.items()}

        sample = noise
        t_seq = shift * np.linspace(1, 0, steps + 1) / (1 + (shift - 1) * np.linspace(1, 0, steps + 1))
        t_pairs = list(zip(t_seq[:-1], t_seq[1:]))

        # tqdm drives both the terminal display and the external callback,
        # so server log and web UI see the same sampler progress.
        from tqdm.auto import tqdm
        iterator = tqdm(t_pairs, desc="TripoSplat Sampling", total=steps)

        for t, t_prev in iterator:
            x_t = {k: v for k, v in sample.items()}
            t_scaled = mx.array([1000 * t], dtype=mx.float32)
            pred_v = self._flow_mlx(x_t, t_scaled, cond)
            if guidance_scale is not None and guidance_scale > 1:
                neg_pred_v = self._flow_mlx(x_t, t_scaled, neg_cond)
                for key in pred_v:
                    pred_v[key] = guidance_scale * pred_v[key] - (guidance_scale - 1) * neg_pred_v[key]
            dt = t - t_prev
            for key in sample:
                sample[key] = sample[key] - pred_v[key] * dt
            # MLX is lazily evaluated — without this the whole sampler graph
            # would only materialize at the torch conversion, and progress
            # callbacks would all fire up-front without real compute behind them.
            mx.eval(*sample.values())
            if callback is not None:
                callback(iterator.n, steps)

        return sample

    def image_to_splat(
        self,
        image: Image.Image,
        cfg: HybridSplatConfig = None,
        show_progress: bool = False,
        callback=None,
    ) -> Tuple[bytes, bytes, bytes, Image.Image]:
        if cfg is None:
            cfg = HybridSplatConfig()

        use_mlx = (self.use_mlx_encoders
                   and self._dinov3_mlx is not None
                   and self._vae_mlx is not None
                   and self._flow_mlx is not None)

        if use_mlx:
            logger.info("Running TripoSplat with MLX encoders + MLX flow model + MPS decoder")
            prepared = self.preprocess_image(image, erode_radius=cfg.erode_radius)
            cond = self.encode_image_mlx(prepared)
            latent = self._sample_latent_mlx(
                cond,
                steps=cfg.steps,
                guidance_scale=cfg.guidance_scale,
                shift=cfg.shift,
                seed=cfg.seed,
                show_progress=show_progress,
                callback=callback,
            )
            # Convert MLX latent to PyTorch for the decoder.
            latent_torch = {
                'latent': torch.from_numpy(np.array(latent['latent'])).to(self.device),
            }
            if 'camera' in latent:
                latent_torch['camera'] = torch.from_numpy(np.array(latent['camera'])).to(self.device)
            gaussian = self._torch_pipeline.decode_latent(latent_torch['latent'], num_gaussians=cfg.num_gaussians)
            from .spz_utils import gaussian_to_spz_bytes
            return gaussian.to_ply_bytes(), gaussian.to_splat_bytes(), gaussian_to_spz_bytes(gaussian), prepared

        gaussian, prepared = self._torch_pipeline.run(
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
        from .spz_utils import gaussian_to_spz_bytes
        return gaussian.to_ply_bytes(), gaussian.to_splat_bytes(), gaussian_to_spz_bytes(gaussian), prepared


def create_hybrid_pipeline(
    triposplat_dir: Union[str, Path],
    device: str = "mps",
    use_mlx_encoders: bool = True,
) -> TripoSplatHybridPipeline:
    triposplat_dir = Path(triposplat_dir)
    return TripoSplatHybridPipeline(
        ckpt_path=triposplat_dir / "diffusion_models" / "triposplat_fp16.safetensors",
        decoder_path=triposplat_dir / "vae" / "triposplat_vae_decoder_fp16.safetensors",
        dinov3_path=triposplat_dir / "clip_vision" / "dino_v3_vit_h.safetensors",
        flux2_vae_encoder_path=triposplat_dir / "vae" / "flux2-vae.safetensors",
        rmbg_path=triposplat_dir / "background_removal" / "birefnet.safetensors",
        device=device,
        use_mlx_encoders=use_mlx_encoders,
    )
