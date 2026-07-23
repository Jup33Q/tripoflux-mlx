"""SPZ export utilities for TripoSplat Gaussian objects."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch


def _import_spz_native():
    """Import the SPZ native module, working around its circular-import issue."""
    spz_dir = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages" / "spz"
    if str(spz_dir) not in sys.path:
        sys.path.insert(0, str(spz_dir))
    import spz as spz_native  # type: ignore
    return spz_native


def gaussian_to_spz_bytes(gaussian) -> bytes:
    """Convert a TripoSplat Gaussian object to SPZ bytes.

    Args:
        gaussian: A `tripoflux.vendor.triposplat.triposplat.Gaussian` instance.

    Returns:
        SPZ file content as bytes.
    """
    spz = _import_spz_native()

    xyz = gaussian.get_xyz.detach().cpu().numpy().astype(np.float32)
    scale = torch.log(gaussian.get_scaling).detach().cpu().numpy().astype(np.float32)
    rotation = (gaussian._rotation + gaussian.rots_bias[None, :]).detach().cpu().numpy().astype(np.float32)
    # Normalize quaternion
    rotation = rotation / np.linalg.norm(rotation, axis=-1, keepdims=True)
    opacity = gaussian.get_opacity.detach().cpu().numpy().astype(np.float32).squeeze(-1)

    # SH DC -> RGB [0, 1]
    C0 = 0.28209479177387814
    f_dc = gaussian._features_dc.detach().cpu().numpy()
    rgb = np.clip(f_dc[:, 0, :] * C0 + 0.5, 0, 1).astype(np.float32)

    splat = spz.GaussianSplat(
        positions=xyz,
        scales=scale,
        rotations=rotation,
        alphas=opacity,
        colors=rgb,
    )
    return splat.to_bytes()


def save_spz(gaussian, path: str) -> None:
    """Save a TripoSplat Gaussian object as an SPZ file."""
    data = gaussian_to_spz_bytes(gaussian)
    with open(path, "wb") as f:
        f.write(data)
