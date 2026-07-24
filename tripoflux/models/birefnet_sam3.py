"""SAM3-based background removal via MLX (mlx-vlm).

Uses the `mlx-community/sam3-8bit` model for open-vocabulary instance
segmentation, then converts the most salient mask into an alpha matte.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image, ImageFilter

logger = logging.getLogger(__name__)


class SAM3BackgroundRemover:
    """Remove image background using SAM3 instance segmentation."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        score_threshold: float = 0.3,
        text_prompt: str = "object",
        local_files_only: bool = False,
    ):
        self.model_path = model_path or "mlx-community/sam3-8bit"
        self.score_threshold = score_threshold
        self.text_prompt = text_prompt
        self.local_files_only = local_files_only
        self._predictor = None

    def _load(self):
        if self._predictor is not None:
            return self._predictor
        try:
            from mlx_vlm.utils import get_model_path, load_model
            from mlx_vlm.models.sam3.generate import Sam3Predictor
            from mlx_vlm.models.sam3.processing_sam3 import Sam3Processor

            model_path = self.model_path
            if self.local_files_only and not Path(str(model_path)).expanduser().exists():
                # Offline: resolve the repo id against the local HF cache only.
                # Raises immediately when the cached snapshot is incomplete
                # instead of retrying the hub for minutes.
                from huggingface_hub import snapshot_download

                model_path = snapshot_download(str(model_path), local_files_only=True)

            path = get_model_path(model_path)
            model = load_model(path)
            processor = Sam3Processor.from_pretrained(str(path))
            self._predictor = Sam3Predictor(model, processor, score_threshold=self.score_threshold)
            logger.info("Loaded SAM3 from %s", self.model_path)
            return self._predictor
        except Exception as exc:
            logger.warning("SAM3 load failed: %s", exc)
            return None

    def remove_background(self, image: Image.Image, erode_radius: int = 1, text_prompt: Optional[str] = None) -> Image.Image:
        """Remove background and return an RGBA image.

        Uses SAM3 to segment the most salient object, then converts the
        binary mask into a soft alpha matte via distance transform.

        ``text_prompt`` overrides the default open-vocabulary prompt; pass the
        generation prompt's subject (e.g. "a tree", "a building") so SAM3
        segments the intended subject instead of whatever looks salient.
        """
        if image.mode != "RGB":
            image = image.convert("RGB")

        predictor = self._load()
        if predictor is None:
            raise RuntimeError("SAM3 is not available")

        result = predictor.predict(image, text_prompt=text_prompt or self.text_prompt)
        masks = result.masks
        if masks is None or len(masks) == 0:
            logger.warning("SAM3 found no objects, returning original image")
            rgba = image.copy()
            rgba.putalpha(Image.new("L", image.size, 255))
            return rgba

        # Pick the highest-score mask
        best_idx = int(np.argmax(result.scores))
        mask = masks[best_idx]  # (H, W) binary or float

        # Convert binary mask to soft alpha matte using distance transform
        mask_np = np.array(mask, dtype=np.float32)
        if mask_np.max() > 1.0:
            mask_np = mask_np / 255.0

        # Simple dilation + blur to soften edges
        alpha = (mask_np * 255).astype(np.uint8)
        alpha_img = Image.fromarray(alpha, mode="L")
        if erode_radius > 0:
            alpha_img = alpha_img.filter(ImageFilter.MinFilter(2 * erode_radius + 1))
        alpha_img = alpha_img.filter(ImageFilter.GaussianBlur(radius=2))

        rgba = image.copy()
        rgba.putalpha(alpha_img)
        return rgba


def create_sam3_remover(
    model_path: Optional[Union[str, Path]] = None,
    score_threshold: float = 0.3,
) -> SAM3BackgroundRemover:
    return SAM3BackgroundRemover(model_path=model_path, score_threshold=score_threshold)
