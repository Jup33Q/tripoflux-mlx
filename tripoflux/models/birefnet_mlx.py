"""BiRefNet background removal with MLX-first routing.

This module provides a unified background-removal interface that prefers an
MLX implementation (SAM3), falling back to CoreML and then PyTorch MPS.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from .birefnet_coreml import BiRefNetCoreML
from .birefnet_sam3 import SAM3BackgroundRemover

logger = logging.getLogger(__name__)


class BiRefNetMLX:
    """MLX-first BiRefNet background remover.

    Backend priority:
    - ``mlx``: SAM3 via mlx-vlm → CoreML → PyTorch MPS
    - ``coreml``: CoreML → PyTorch MPS
    - ``mps``: PyTorch MPS only
    """

    def __init__(
        self,
        triposplat_dir: Union[str, Path],
        backend: str = "mlx",
        device: str = "mps",
        sam3_model: Optional[str] = None,
    ):
        self.triposplat_dir = Path(triposplat_dir)
        self.backend = backend
        self.device = device
        self._sam3: Optional[SAM3BackgroundRemover] = None
        self._coreml = BiRefNetCoreML(
            coreml_path=self.triposplat_dir / "birefnet.mlpackage",
            fallback_torch_path=self.triposplat_dir / "background_removal" / "birefnet.safetensors",
            device=device,
        )
        if backend == "mlx":
            try:
                self._sam3 = SAM3BackgroundRemover(model_path=sam3_model)
                logger.info("SAM3 background remover initialized")
            except Exception as exc:
                logger.warning("SAM3 init failed, will use fallback: %s", exc)
                self._sam3 = None

    def remove_background(self, image: Image.Image) -> Image.Image:
        """Remove the background and return an RGBA image."""
        if self.backend == "mlx" and self._sam3 is not None:
            try:
                return self._sam3.remove_background(image)
            except Exception as exc:
                logger.warning("SAM3 inference failed, falling back: %s", exc)
        return self._coreml.remove_background(image)

    @property
    def backend_name(self) -> str:
        if self.backend == "mlx":
            if self._sam3 is not None:
                return "mlx(sam3)"
            return "mlx(fallback:coreml/mps)"
        return self.backend


def create_birefnet_remover(
    triposplat_dir: Union[str, Path],
    backend: str = "mlx",
    device: str = "mps",
    sam3_model: Optional[str] = None,
) -> BiRefNetMLX:
    return BiRefNetMLX(
        triposplat_dir=triposplat_dir,
        backend=backend,
        device=device,
        sam3_model=sam3_model,
    )
