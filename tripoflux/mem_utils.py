"""Memory hygiene helpers for the long-lived server process.

Apple Silicon shares memory between CPU and GPU: allocator caches (MLX Metal
cache, PyTorch MPS cache) plus every job's results quickly push the machine
into swap, and generation slows by an order of magnitude once that happens.
These helpers release cached-but-unreferenced memory between jobs — models
stay resident, only caches and explicit unloads are freed.
"""

from __future__ import annotations

import gc
import logging

logger = logging.getLogger(__name__)


def release_gpu_caches() -> None:
    """Return unused GPU/allocator memory to the OS.

    Runs Python GC, then clears MLX's Metal buffer cache and PyTorch's MPS
    cache. Tensors and models still referenced elsewhere are unaffected.
    """
    gc.collect()
    try:
        import mlx.core as mx

        mx.clear_cache()
    except Exception:
        pass
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
