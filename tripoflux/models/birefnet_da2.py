"""DA2 (Depth Anything V2) background removal.

Uses depth estimation to separate foreground from background. Supports both
PyTorch (transformers pipeline) and CoreML backends.
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
        use_coreml: bool = False,
    ):
        """
        Args:
            model_name: HuggingFace model ID for PyTorch backend. Options:
                - depth-anything/Depth-Anything-V2-Small
                - depth-anything/Depth-Anything-V2-Base
                - depth-anything/Depth-Anything-V2-Large
            device: PyTorch device (mps, cpu, cuda).
            foreground_percentile: Depth percentile threshold for foreground.
            use_coreml: Use Apple's official CoreML model (small only).
        """
        self.model_name = model_name
        self.device = torch.device(device)
        self.foreground_percentile = foreground_percentile
        self.use_coreml = use_coreml
        self._pipe = None
        self._coreml_model = None

    def _load(self):
        if self.use_coreml:
            return self._load_coreml()
        return self._load_pytorch()

    def _load_pytorch(self):
        if self._pipe is not None:
            return self._pipe
        try:
            from transformers import pipeline

            self._pipe = pipeline(
                task="depth-estimation",
                model=self.model_name,
                device=self.device,
            )
            logger.info("Loaded DA2 PyTorch model: %s", self.model_name)
            return self._pipe
        except Exception as exc:
            logger.warning("DA2 PyTorch load failed: %s", exc)
            return None

    def _load_coreml(self):
        if self._coreml_model is not None:
            return self._coreml_model
        try:
            from huggingface_hub import snapshot_download
            import coremltools as ct

            model_path = snapshot_download("apple/coreml-depth-anything-v2-small")
            mlpackage = list(Path(model_path).glob("*.mlpackage"))[0]
            self._coreml_model = ct.models.MLModel(str(mlpackage))
            logger.info("Loaded DA2 CoreML model: apple/coreml-depth-anything-v2-small")
            return self._coreml_model
        except Exception as exc:
            logger.warning("DA2 CoreML load failed: %s", exc)
            return None

    def _depth_to_alpha(self, depth_map: np.ndarray) -> np.ndarray:
        """Convert a depth map to an alpha matte using thresholding."""
        d_min, d_max = depth_map.min(), depth_map.max()
        if d_max - d_min < 1e-8:
            return np.ones_like(depth_map, dtype=np.float32)
        norm = (depth_map - d_min) / (d_max - d_min)

        threshold = np.percentile(norm, 100.0 - self.foreground_percentile)
        alpha = np.clip((norm - threshold) / max(1e-8, 1.0 - threshold), 0, 1)

        alpha_img = Image.fromarray((alpha * 255).astype(np.uint8), mode="L")
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=3))
        return np.array(alpha_img, dtype=np.float32) / 255.0

    def _preprocess_coreml(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for CoreML DA2 model."""
        # CoreML DA2 expects (1, 3, 518, 518) float32 in [0, 1]
        img = image.resize((518, 518), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        return arr[None, ...]  # Add batch dim

    def remove_background(self, image: Image.Image) -> Image.Image:
        """Remove background and return an RGBA image."""
        if image.mode != "RGB":
            image = image.convert("RGB")

        model = self._load()
        if model is None:
            raise RuntimeError("DA2 model is not available")

        if self.use_coreml:
            # CoreML inference
            input_arr = self._preprocess_coreml(image)
            spec = model.get_spec()
            input_name = spec.description.input[0].name
            output_name = spec.description.output[0].name
            result = model.predict({input_name: input_arr})
            depth_map = np.asarray(result[output_name]).squeeze()
            # Resize depth map back to original size
            depth_img = Image.fromarray(depth_map).resize(image.size, Image.LANCZOS)
            depth_map = np.array(depth_img, dtype=np.float32)
        else:
            # PyTorch inference
            result = model(image)
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
    use_coreml: bool = False,
) -> DA2BackgroundRemover:
    return DA2BackgroundRemover(
        model_name=model_name,
        device=device,
        foreground_percentile=foreground_percentile,
        use_coreml=use_coreml,
    )
