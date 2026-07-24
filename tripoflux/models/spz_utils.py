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

    # Same frame as the ply/splat exports (TripoSplat's default transform).
    xyz, rotation = gaussian._transformed_xyz_rot()
    # SPZ expects log-scales.
    scale = torch.log(gaussian.get_scaling).detach().cpu().numpy().astype(np.float32)
    # SPZ expects opacity as inverse-sigmoid (logit), NOT linear alpha —
    # feeding [0,1] values makes a viewer's sigmoid squash everything to a
    # faint 0.5..0.73 ghost.
    opacity = gaussian.get_opacity.detach().cpu().numpy().astype(np.float64).squeeze(-1)
    opacity = np.clip(opacity, 1e-6, 1 - 1e-6)
    alphas = np.log(opacity / (1.0 - opacity)).astype(np.float32)
    # SPZ expects raw SH0 (f_dc) coefficients, NOT converted sRGB — feeding
    # [0,1] rgb makes every point render as the same washed-out gray.
    colors = gaussian._features_dc.detach().cpu().numpy()[:, 0, :].astype(np.float32)

    splat = spz.GaussianSplat(
        positions=xyz.astype(np.float32),
        scales=scale,
        rotations=rotation.astype(np.float32),  # (w, x, y, z), SPZ's order
        alphas=alphas,
        colors=colors,
    )
    # Our frame follows the standard 3DGS PLY convention; the writer converts
    # it to SPZ's native RUB storage.
    return splat.to_bytes(from_coordinate_system=spz.CoordinateSystem.RDF)


def save_spz(gaussian, path: str) -> None:
    """Save a TripoSplat Gaussian object as an SPZ file."""
    data = gaussian_to_spz_bytes(gaussian)
    with open(path, "wb") as f:
        f.write(data)


# ---------------------------------------------------------------------------
# SPZ → derived formats (preview .splat and .ply are both exported FROM the
# canonical SPZ bytes, so what the viewport renders is exactly what the SPZ
# download contains)
# ---------------------------------------------------------------------------

def _load_spz_arrays(spz_bytes: bytes):
    """Decode SPZ bytes to raw arrays in TripoSplat's saved frame.

    The SPZ writer stores data in RUB after converting from our (PLY/RDF)
    frame — a 180° rotation about X — so positions and quaternions are
    converted back here. Returns (xyz, scales_log, quat, alphas_logit, sh0).
    """
    spz = _import_spz_native()
    s = spz.GaussianSplat.from_bytes(spz_bytes)

    xyz = s.positions * np.array([1.0, -1.0, -1.0], dtype=np.float32)
    # Frame conjugation by the 180°-about-X rotation: (w,x,y,z) -> (w,x,-y,-z)
    quat = s.rotations * np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float32)
    quat = quat / np.linalg.norm(quat, axis=-1, keepdims=True)
    return xyz, s.scales, quat.astype(np.float32), s.alphas, s.colors


def spz_bytes_to_splat_bytes(spz_bytes: bytes) -> bytes:
    """Convert SPZ bytes to the 32-byte-per-splat preview format.

    Layout matches `Gaussian.to_splat_bytes`: xyz(f32x3) + scale(f32x3,
    linear) + rgba(u8x4) + rot(u8x4, wxyz), sorted by descending
    opacity * volume.
    """
    C0 = 0.28209479177387814
    xyz, scales_log, quat, alphas, sh0 = _load_spz_arrays(spz_bytes)

    scale = np.exp(scales_log).astype(np.float32)
    opacity = 1.0 / (1.0 + np.exp(-alphas.astype(np.float64)))
    rgb = np.clip((sh0 * C0 + 0.5) * 255, 0, 255).astype(np.uint8)
    alpha = np.clip(opacity * 255, 0, 255).astype(np.uint8).reshape(-1, 1)
    rgba = np.concatenate([rgb, alpha], axis=1)
    rot_u8 = np.clip(quat * 128 + 128, 0, 255).astype(np.uint8)

    order = np.argsort(-opacity * np.prod(scale, axis=-1))
    xyz, scale, rgba, rot_u8 = xyz[order], scale[order], rgba[order], rot_u8[order]
    data = np.concatenate([
        xyz.astype(np.float32).view(np.uint8).reshape(-1, 12),
        scale.astype(np.float32).view(np.uint8).reshape(-1, 12),
        rgba.reshape(-1, 4),
        rot_u8.reshape(-1, 4),
    ], axis=1).reshape(-1)
    return data.tobytes()


def spz_bytes_to_ply_bytes(spz_bytes: bytes) -> bytes:
    """Convert SPZ bytes to a binary 3DGS .ply (same layout as
    `Gaussian.to_ply_bytes`, including uchar red/green/blue vertex colors)."""
    from ..vendor.triposplat.triposplat import _binary_ply_bytes

    C0 = 0.28209479177387814
    xyz, scales_log, quat, alphas, sh0 = _load_spz_arrays(spz_bytes)
    n = xyz.shape[0]

    attrs = (["x", "y", "z", "nx", "ny", "nz"]
             + [f"f_dc_{i}" for i in range(3)]
             + ["opacity"]
             + [f"scale_{i}" for i in range(3)]
             + [f"rot_{i}" for i in range(4)])
    dtype_full = [(a, "f4") for a in attrs] + [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    rgb = (np.clip(sh0 * C0 + 0.5, 0, 1) * 255).round().astype(np.uint8)

    cols = np.concatenate(
        [xyz, np.zeros_like(xyz), sh0, alphas.reshape(-1, 1), scales_log, quat, rgb],
        axis=1,
    )
    elements = np.empty(n, dtype=dtype_full)
    elements[:] = list(map(tuple, cols))
    return _binary_ply_bytes(elements, dtype_full)
