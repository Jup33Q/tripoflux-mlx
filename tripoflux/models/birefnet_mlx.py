"""BiRefNet background removal with MLX-first routing.

This module provides a unified background-removal interface that prefers an
MLX implementation (SAM3 or DA2), falling back to CoreML and then PyTorch MPS.
"""

from __future__ import annotations

import logging
import os
import socket
from pathlib import Path
from typing import Optional, Union
from urllib.parse import urlparse

from PIL import Image

from .birefnet_coreml import BiRefNetCoreML
from .birefnet_da2 import DA2BackgroundRemover
from .birefnet_sam3 import SAM3BackgroundRemover

logger = logging.getLogger(__name__)


def _hub_available(timeout: float = 2.0) -> bool:
    """Quick reachability probe for the HF hub (or HF_ENDPOINT mirror).

    When the hub is unreachable, hub-backed model loads would otherwise burn
    minutes in connection retries before falling back to the local cache.
    Uses httpx first so proxy env vars are honored (a raw socket bypasses
    HTTP proxies and reports false negatives); falls back to a TCP probe.
    """
    endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    try:
        import httpx

        try:
            with httpx.Client(timeout=timeout, follow_redirects=True) as client:
                client.head(endpoint)
            return True
        except httpx.HTTPError:
            pass
    except ImportError:
        pass
    host = urlparse(endpoint).hostname or "huggingface.co"
    try:
        with socket.create_connection((host, 443), timeout=timeout):
            return True
    except OSError:
        return False


class BiRefNetMLX:
    """MLX-first BiRefNet background remover.

    Backend priority:
    - ``mlx``: SAM3 via mlx-vlm → DA2 → CoreML → PyTorch MPS
    - ``da2``: DA2 depth-based removal → CoreML → PyTorch MPS
    - ``coreml``: CoreML → PyTorch MPS
    - ``mps``: PyTorch MPS only
    """

    def __init__(
        self,
        triposplat_dir: Union[str, Path],
        backend: str = "mlx",
        device: str = "mps",
        sam3_model: Optional[str] = None,
        da2_model: Optional[str] = None,
        da2_coreml: bool = False,
    ):
        self.triposplat_dir = Path(triposplat_dir)
        self.backend = backend
        self.device = device
        self._sam3: Optional[SAM3BackgroundRemover] = None
        self._da2: Optional[DA2BackgroundRemover] = None
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
        if backend in ("mlx", "da2"):
            try:
                self._da2 = DA2BackgroundRemover(
                    model_name=da2_model or "depth-anything/Depth-Anything-V2-Base-hf",
                    device=device,
                    use_coreml=da2_coreml,
                )
                logger.info("DA2 background remover initialized (coreml=%s)", da2_coreml)
            except Exception as exc:
                logger.warning("DA2 init failed, will use fallback: %s", exc)
                self._da2 = None

    @staticmethod
    def _subject_prompt(prompt: Optional[str]) -> Optional[str]:
        """Extract a short subject phrase for SAM3 from a generation prompt.

        SAM3's open-vocabulary detection works best with a short noun phrase,
        so use the first comma-separated clause ("a tree, low-poly style" →
        "a tree").
        """
        if not prompt:
            return None
        subject = prompt.split(",")[0].strip()
        return subject or None

    def remove_background(self, image: Image.Image, prompt: Optional[str] = None) -> Image.Image:
        """Remove the background and return an RGBA image.

        ``prompt`` is the generation prompt; its subject clause is forwarded
        to SAM3 so prompt-aware segmentation keeps the intended subject
        (buildings, trees, ...) instead of whatever looks salient.
        """
        if not _hub_available():
            # Offline: restrict hub-backed removers to the local cache so a
            # missing/incomplete snapshot fails in milliseconds instead of
            # retrying the network for minutes.
            if self._sam3 is not None:
                self._sam3.local_files_only = True
            if self._da2 is not None:
                self._da2.local_files_only = True
        if self.backend == "mlx" and self._sam3 is not None:
            try:
                return self._sam3.remove_background(
                    image, text_prompt=self._subject_prompt(prompt)
                )
            except Exception as exc:
                logger.warning("SAM3 inference failed, falling back: %s", exc)
        if self.backend in ("mlx", "da2") and self._da2 is not None:
            try:
                return self._da2.remove_background(image)
            except Exception as exc:
                logger.warning("DA2 inference failed, falling back: %s", exc)
        return self._coreml.remove_background(image)

    @property
    def backend_name(self) -> str:
        if self.backend == "mlx":
            if self._sam3 is not None:
                return "mlx(sam3)"
            if self._da2 is not None:
                return "mlx(da2)"
            return "mlx(fallback:coreml/mps)"
        return self.backend


def create_birefnet_remover(
    triposplat_dir: Union[str, Path],
    backend: str = "mlx",
    device: str = "mps",
    sam3_model: Optional[str] = None,
    da2_model: Optional[str] = None,
    da2_coreml: bool = False,
) -> BiRefNetMLX:
    return BiRefNetMLX(
        triposplat_dir=triposplat_dir,
        backend=backend,
        device=device,
        sam3_model=sam3_model,
        da2_model=da2_model,
        da2_coreml=da2_coreml,
    )
