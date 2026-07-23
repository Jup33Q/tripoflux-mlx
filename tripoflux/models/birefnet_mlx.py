"""BiRefNet background removal with MLX-first routing.

This module provides a unified background-removal interface that prefers an
MLX implementation, falling back to CoreML and then PyTorch MPS.

Note: a full MLX port of BiRefNet (Swin-L + deformable ASPP decoder) is
non-trivial and is planned for the post-MVP Phase 7. For now the MLX
backend delegates to the fastest available implementation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

from PIL import Image

from .birefnet_coreml import BiRefNetCoreML

logger = logging.getLogger(__name__)


class BiRefNetMLX:
    """MLX-first BiRefNet background remover.

    Currently the MLX path falls back to CoreML (if a converted model
    exists) or PyTorch MPS. A native MLX implementation is scheduled for
    Phase 7.
    """

    def __init__(
        self,
        triposplat_dir: Union[str, Path],
        backend: str = "mlx",
        device: str = "mps",
    ):
        self.triposplat_dir = Path(triposplat_dir)
        self.backend = backend
        self.device = device
        self._coreml = BiRefNetCoreML(
            coreml_path=self.triposplat_dir / "birefnet.mlpackage",
            fallback_torch_path=self.triposplat_dir / "background_removal" / "birefnet.safetensors",
            device=device,
        )

    def remove_background(self, image: Image.Image) -> Image.Image:
        """Remove the background and return an RGBA image."""
        # TODO(Phase 7): route to native MLX implementation here when ready.
        return self._coreml.remove_background(image)

    @property
    def backend_name(self) -> str:
        if self.backend == "mlx":
            return "mlx(fallback:coreml/mps)"
        return self.backend


def create_birefnet_remover(
    triposplat_dir: Union[str, Path],
    backend: str = "mlx",
    device: str = "mps",
) -> BiRefNetMLX:
    return BiRefNetMLX(triposplat_dir=triposplat_dir, backend=backend, device=device)
