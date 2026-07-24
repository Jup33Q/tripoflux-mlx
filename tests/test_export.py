"""Round-trip tests for PLY / SPZ splat exports."""

import numpy as np
import torch

from tripoflux.models.spz_utils import _import_spz_native, gaussian_to_spz_bytes
from tripoflux.vendor.triposplat.triposplat import Gaussian

C0 = 0.28209479177387814


def _make_gaussian(n: int = 16) -> Gaussian:
    torch.manual_seed(0)
    g = Gaussian(aabb=[-0.5, -0.5, -0.5, 1.0, 1.0, 1.0], device="cpu")
    g._xyz = torch.rand(n, 3)
    g._scaling = torch.full((n, 3), -4.0)
    g._rotation = torch.zeros(n, 4)  # + rots_bias -> identity quat (w=1)
    g._opacity = torch.zeros(n, 1)  # sigmoid(0 + logit(0.1)) = 0.1
    g._features_dc = torch.rand(n, 1, 3) * 2 - 1
    return g


def test_ply_has_vertex_colors():
    n = 16
    g = _make_gaussian(n)
    data = g.to_ply_bytes()
    header_blob, payload = data.split(b"end_header\n", 1)
    header = header_blob.decode("ascii")

    # Generic viewers read uchar red/green/blue, not SH f_dc.
    for chan in ("red", "green", "blue"):
        assert f"property uchar {chan}" in header
    assert "property float f_dc_0" in header

    # 17 float props (xyz, normals, f_dc x3, opacity, scale x3, rot x4) + 3 u1
    assert len(payload) == n * (17 * 4 + 3)

    # Spot-check vertex 0's RGB against its f_dc-derived color.
    f_dc = g._features_dc.detach().transpose(1, 2).flatten(start_dim=1).numpy()
    expected = (np.clip(f_dc * C0 + 0.5, 0, 1) * 255).round().astype(np.uint8)
    rec = 17 * 4
    got = np.frombuffer(payload[68:71], dtype=np.uint8)  # first vertex tail
    np.testing.assert_array_equal(got, expected[0])
    assert len(payload) == n * (rec + 3)


def test_spz_roundtrip_preserves_opacity_color_scale():
    n = 16
    g = _make_gaussian(n)
    data = gaussian_to_spz_bytes(g)

    spz = _import_spz_native()
    s = spz.GaussianSplat.from_bytes(data)
    assert s.num_points == n

    # Opacity: stored as logit; a viewer's sigmoid must recover it.
    opacity = g.get_opacity.numpy().squeeze(-1)
    np.testing.assert_allclose(1 / (1 + np.exp(-s.alphas)), opacity, atol=0.05)

    # Color: stored as raw SH0; a viewer's 0.5 + C0*sh0 must recover sRGB.
    f_dc = g._features_dc.numpy()[:, 0, :]
    np.testing.assert_allclose(0.5 + C0 * s.colors, 0.5 + C0 * f_dc, atol=0.05)

    # Scale: stored as log-scale.
    np.testing.assert_allclose(np.exp(s.scales), g.get_scaling.numpy(), rtol=0.1)

    # Positions: TripoSplat's default transform, then RDF->RUB on save
    # (rotate 180 deg about X => y, z negated).
    xyz, _ = g._transformed_xyz_rot()
    np.testing.assert_allclose(s.positions, xyz * [1, -1, -1], atol=1e-2)


def _parse_splat_records(data: bytes, n: int):
    rec = np.frombuffer(data, dtype=np.uint8).reshape(n, 32)
    xyz = rec[:, 0:12].copy().view(np.float32).reshape(n, 3)
    scale = rec[:, 12:24].copy().view(np.float32).reshape(n, 3)
    rgba = rec[:, 24:28]
    return xyz, scale, rgba


def test_spz_derived_splat_matches_direct_export():
    """The SPZ-derived preview .splat must match the direct gaussian export,
    modulo SPZ quantization."""
    from tripoflux.models.spz_utils import spz_bytes_to_splat_bytes

    n = 64
    g = _make_gaussian(n)
    direct = g.to_splat_bytes()
    derived = spz_bytes_to_splat_bytes(gaussian_to_spz_bytes(g))
    assert len(derived) == n * 32

    d_xyz, d_scale, d_rgba = _parse_splat_records(direct, n)
    s_xyz, s_scale, s_rgba = _parse_splat_records(derived, n)

    # Compare as sets (record sort order is not stable for equal keys).
    key = lambda a: np.lexsort((a[:, 2], a[:, 1], a[:, 0]))
    d_xyz, s_xyz = d_xyz[key(d_xyz)], s_xyz[key(s_xyz)]
    np.testing.assert_allclose(s_xyz, d_xyz, atol=2e-3)          # 24-bit pos
    np.testing.assert_allclose(s_scale, d_scale, rtol=0.05)      # 1/16 log-scale
    assert np.abs(s_rgba[:, :3].astype(int) - d_rgba[:, :3].astype(int)).max() <= 8
    assert np.abs(s_rgba[:, 3].astype(int) - d_rgba[:, 3].astype(int)).max() <= 3


def test_spz_derived_ply_matches_layout_and_values():
    """The SPZ-derived .ply must carry the standard 3DGS attributes plus
    uchar vertex colors, with values close to the gaussian source."""
    from tripoflux.models.spz_utils import spz_bytes_to_ply_bytes

    n = 16
    g = _make_gaussian(n)
    data = spz_bytes_to_ply_bytes(gaussian_to_spz_bytes(g))
    header_blob, payload = data.split(b"end_header\n", 1)
    header = header_blob.decode("ascii")
    for prop in ("float x", "float f_dc_0", "float opacity", "float scale_0",
                 "float rot_0", "uchar red", "uchar green", "uchar blue"):
        assert f"property {prop}" in header
    assert len(payload) == n * (17 * 4 + 3)

    # Vertex 0: f_dc == SH0 color, opacity == logit, scale == log-scale.
    f32 = np.frombuffer(payload[: 17 * 4], dtype=np.float32)
    f_dc = g._features_dc.numpy()[:, 0, :]
    # f_dc_0..2 are props 6..8; find the source point by position match.
    xyz, _ = g._transformed_xyz_rot()
    idx = int(np.argmin(np.abs(xyz - f32[0:3]).sum(axis=1)))
    np.testing.assert_allclose(f32[6:9], f_dc[idx], atol=0.05)
    opacity = g.get_opacity.numpy().squeeze(-1)[idx]
    np.testing.assert_allclose(1 / (1 + np.exp(-f32[9])), opacity, atol=0.05)
    np.testing.assert_allclose(np.exp(f32[10:13]), g.get_scaling.numpy()[idx], rtol=0.1)
