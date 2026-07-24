"""FLUX.2-klein-9B image generation via MLX (mflux).

Uses the `mflux` package, which provides native MLX implementations of
FLUX.2 models on Apple Silicon. Falls back to the Hugging Face diffusers
pipeline on MPS when `mflux` is unavailable or fails.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from PIL import Image

logger = logging.getLogger(__name__)

StepCallback = Callable[[int, int], None]  # (completed_steps, total_steps)


class _MfluxStepCallback:
    """Adapter implementing the mflux InLoopCallback protocol.

    A single instance is registered on the model once; the active per-call
    callback is swapped in via ``fn`` so repeated generations do not
    accumulate registrations.
    """

    def __init__(self) -> None:
        self.fn: Optional[StepCallback] = None

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps) -> None:
        if self.fn is None:
            return
        total = int(getattr(config, "num_inference_steps", 0) or 0)
        try:
            self.fn(int(t) + 1, total)
        except Exception:
            pass


@dataclass(frozen=True)
class FluxGenerationConfig:
    prompt: str
    width: int = 1024
    height: int = 1024
    num_inference_steps: int = 4
    guidance_scale: float = 1.0
    seed: int = 42
    quantize: int = 8  # 4 or 8 for mflux
    negative_prompt: Optional[str] = None


class FluxKleinGenerator:
    """Generate images with FLUX.2-klein-9B using MLX when possible."""

    def __init__(
        self,
        backend: str = "mlx",
        model_name: str = "flux2-klein-9b",
        quantize: int = 8,
        torch_device: str = "mps",
    ):
        self.backend = backend
        self.model_name = model_name
        self.quantize = quantize
        self.torch_device = torch_device
        self._mflux_model = None
        self._mflux_step_cb: Optional[_MfluxStepCallback] = None
        self._diffusers_pipe = None

    # ------------------------------------------------------------------
    # MLX path (mflux)
    # ------------------------------------------------------------------
    def _load_mflux(self):
        if self._mflux_model is not None:
            return self._mflux_model
        try:
            from mflux.models.common.config import ModelConfig
            from mflux.models.flux2 import Flux2Klein

            if self.model_name == "flux2-klein-9b":
                model_config = ModelConfig.flux2_klein_9b()
            elif self.model_name == "flux2-klein-4b":
                model_config = ModelConfig.flux2_klein_4b()
            else:
                raise ValueError(f"unsupported mflux model name: {self.model_name}")

            self._mflux_model = Flux2Klein(
                model_config=model_config,
                quantize=self.quantize,
            )
            self._mflux_step_cb = _MfluxStepCallback()
            callbacks = getattr(self._mflux_model, "callbacks", None)
            if callbacks is not None:
                callbacks.register(self._mflux_step_cb)
            else:
                self._mflux_step_cb = None
            logger.info("Loaded FLUX.2-klein-9B via mflux (MLX)")
            return self._mflux_model
        except Exception as exc:  # pragma: no cover - environment specific
            logger.warning("mflux load failed: %s", exc)
            return None

    def _generate_mflux(
        self,
        cfg: FluxGenerationConfig,
        step_callback: Optional[StepCallback] = None,
    ) -> Optional[Image.Image]:
        model = self._load_mflux()
        if model is None:
            return None
        try:
            if cfg.negative_prompt:
                logger.info("FLUX.2-klein does not use negative_prompt; ignoring it")
            if self._mflux_step_cb is not None:
                self._mflux_step_cb.fn = step_callback
            kwargs = dict(
                seed=cfg.seed,
                prompt=cfg.prompt,
                num_inference_steps=cfg.num_inference_steps,
                width=cfg.width,
                height=cfg.height,
                guidance=cfg.guidance_scale,
            )
            result = model.generate_image(**kwargs)
            # mflux returns a GeneratedImage wrapper; unwrap the PIL image.
            if hasattr(result, "image"):
                result = result.image
            if hasattr(result, "convert"):
                return result.convert("RGB")
            return result
        except Exception as exc:  # pragma: no cover - runtime failure
            logger.warning("mflux generation failed: %s", exc)
            self._last_mflux_error = exc
            return None
        finally:
            if self._mflux_step_cb is not None:
                self._mflux_step_cb.fn = None

    # ------------------------------------------------------------------
    # Diffusers / MPS fallback
    # ------------------------------------------------------------------
    def _load_diffusers(self):
        if self._diffusers_pipe is not None:
            return self._diffusers_pipe
        try:
            import torch
            from diffusers import Flux2KleinPipeline

            pipe = Flux2KleinPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-klein-9B",
                torch_dtype=torch.bfloat16,
            )
            pipe.to(self.torch_device)
            if hasattr(pipe, "enable_model_cpu_offload"):
                pipe.enable_model_cpu_offload()
            self._diffusers_pipe = pipe
            logger.info("Loaded FLUX.2-klein-9B via diffusers on %s", self.torch_device)
            return pipe
        except Exception as exc:  # pragma: no cover - environment specific
            logger.warning("diffusers load failed: %s", exc)
            return None

    def _generate_diffusers(
        self,
        cfg: FluxGenerationConfig,
        step_callback: Optional[StepCallback] = None,
    ) -> Optional[Image.Image]:
        pipe = self._load_diffusers()
        if pipe is None:
            return None
        try:
            import torch

            generator = torch.Generator(device=self.torch_device).manual_seed(cfg.seed)
            if cfg.negative_prompt:
                logger.info("FLUX.2-klein does not use negative_prompt; ignoring it")
            kwargs = dict(
                prompt=cfg.prompt,
                height=cfg.height,
                width=cfg.width,
                guidance_scale=cfg.guidance_scale,
                num_inference_steps=cfg.num_inference_steps,
                generator=generator,
            )
            if step_callback is not None:
                def _on_step_end(_pipe, step_index, _timestep, cb_kwargs):
                    step_callback(step_index + 1, cfg.num_inference_steps)
                    return cb_kwargs

                kwargs["callback_on_step_end"] = _on_step_end
            image = pipe(**kwargs).images[0]
            return image.convert("RGB")
        except Exception as exc:  # pragma: no cover - runtime failure
            logger.warning("diffusers generation failed: %s", exc)
            self._last_diffusers_error = exc
            return None

    # ------------------------------------------------------------------
    # Memory management
    # ------------------------------------------------------------------
    def unload(self) -> None:
        """Drop every loaded model (used for temporary per-call generators)."""
        self._mflux_model = None
        self._mflux_step_cb = None
        self._diffusers_pipe = None

    def unload_diffusers(self) -> None:
        """Drop the diffusers fallback pipeline (~34 GB) if it was loaded.

        Only meaningful when the primary backend is mlx: a future mflux
        failure would simply reload it.
        """
        if self._diffusers_pipe is not None:
            logger.info("Unloading diffusers fallback pipeline to free memory")
            self._diffusers_pipe = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def generate(
        self,
        cfg: FluxGenerationConfig,
        step_callback: Optional[StepCallback] = None,
    ) -> Image.Image:
        """Generate an image, preferring MLX, falling back to diffusers on MPS."""
        self._last_mflux_error = None
        self._last_diffusers_error = None
        if self.backend == "mlx":
            img = self._generate_mflux(cfg, step_callback=step_callback)
            if img is not None:
                return img
            logger.info("Falling back to diffusers/MPS for FLUX generation")

        img = self._generate_diffusers(cfg, step_callback=step_callback)
        if img is None:
            parts = []
            if self._last_mflux_error is not None:
                parts.append(f"mflux: {self._last_mflux_error}")
            if self._last_diffusers_error is not None:
                parts.append(f"diffusers: {self._last_diffusers_error}")
            detail = "; ".join(parts) if parts else "unknown error"
            raise RuntimeError(f"FLUX.2-klein-9B generation failed on all backends: {detail}")
        return img

    def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        seed: int = 42,
        negative_prompt: Optional[str] = None,
        step_callback: Optional[StepCallback] = None,
    ) -> Image.Image:
        return self.generate(
            FluxGenerationConfig(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                negative_prompt=negative_prompt,
            ),
            step_callback=step_callback,
        )


def create_flux_generator(backend: str = "mlx", quantize: int = 8) -> FluxKleinGenerator:
    return FluxKleinGenerator(backend=backend, quantize=quantize)
