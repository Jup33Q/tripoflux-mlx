"""DA2 (Depth Anything V2) background removal.

Uses depth estimation to separate foreground from background. This is a
lightweight alternative to BiRefNet/SAM3 when a dedicated segmentation
model is not available.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


class DA2BackgroundRemover:
    """Remove image background using Depth Anything V2 depth estimation."""

    def __init__(
        self,
        model_name: str = "depth-anything/Depth-Anything-V2-Base",
        device: str = "mps",
        foreground_percentile: float = 30.0,
    ):
        """
        Args:
            model_name: HuggingFace model ID. Options:
                - depth-anything/Depth-Anything-V2-Small
                - depth-anything/Depth-Anything-V2-Base
                - depth-anything/Depth-Anything-V2-Large
            device: PyTorch device (mps, cpu, cuda).
            foreground_percentile: Depth percentile threshold for foreground.
                Lower values keep more of the image as foreground.
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.foreground_percentile = foreground_percentile
        self._pipe = None

    def _load(self):
        if self._pipe is not None:
            return self._pipe
        try:
            from transformers import pipeline

            self._pipe = pipeline(
                task="depth-estimation",
                model=self.model_name,
                device=self.device,
            )
            logger.info("Loaded DA2 model: %s", self.model_name)
            return self._pipe
        except Exception as exc:
            logger.warning("DA2 load failed: %s", exc)
            return None

    def _depth_to_alpha(self, depth_map: np.ndarray) -> np.ndarray:
        """Convert a depth map to an alpha matte using thresholding.

        Depth Anything outputs disparity (inverse depth). Closer objects
        have higher values. We normalize and threshold to get a soft mask.
        """
        # Normalize to [0, 1]
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min < 1e-8:
            return np.ones_like(depth_map, dtype=np.float32)
        norm = (depth_map - d_min) / (d_max - d_min)

        # Threshold: keep the closest X% as foreground
        threshold = np.percentile(norm, 100.0 - self.foreground_percentile)
        alpha = np.clip((norm - threshold) / max(1e-8, 1.0 - threshold), 0, 1)

        # Smooth edges
        alpha_img = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=3))
        return np.array(alpha_img, dtype=np.float32) / 255.0

    def remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background and return an RGBA image."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        pipe = self._load()
        if pipe is None:
            raise RuntimeError("DA2 model is not available")

        result = pipe(image)
        depth_map = np.array(result["depth"], dtype=np.float32)

        alpha = self._depth_to_alpha(depth_map)
        alpha_img = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")

        rgba = image.copy()
        rgba.putalpha(alpha_img)
        return rgba


def create_da2_remover(
    model_name: str = "depth-anything/Depth-Anything-V2-Base",
    device: str = "mps",
    foreground_percentile: float = 30.0,
) -> DA2BackgroundRemover:
    return DA2BackgroundRemover(
        model_name=model_name,
        device=device,
        foreground_percentile=foreground_percentile,
    )
