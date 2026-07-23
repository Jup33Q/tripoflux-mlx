"""Tests for the TripoFlux MLX pipeline.

These tests are intentionally light-weight: they verify that each stage
produces output of the expected shape/type without downloading large
models or running expensive generation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from tripoflux.models.flux_klein_mlx import FluxGenerationConfig, FluxKleinGenerator
from tripoflux.models.triposplat_wrapped import SplatGenerationConfig, TripoSplatGenerator
from tripoflux.pipeline import PipelineConfig, TripoFluxPipeline


def test_flux_config_defaults():
    cfg = FluxGenerationConfig(prompt="test")
    assert cfg.width == 1024
    assert cfg.height == 1024
    assert cfg.num_inference_steps == 4
    assert cfg.guidance_scale == 1.0
    assert cfg.seed == 42


def test_splat_config_defaults():
    cfg = SplatGenerationConfig()
    assert cfg.num_gaussians == 262144
    assert cfg.steps == 28
    assert cfg.guidance_scale == 3.0


def test_pipeline_config_defaults(tmp_path):
    cfg = PipelineConfig(triposplat_dir=tmp_path)
    assert cfg.flux_backend == "mlx"
    assert cfg.birefnet_backend == "mlx"
    assert cfg.triposplat_backend == "mps"
    assert cfg.num_gaussians == 262144


@patch("tripoflux.models.flux_klein_mlx.FluxKleinGenerator.generate")
def test_flux_generator_interface(mock_generate):
    img = Image.new("RGB", (1024, 1024), color="red")
    mock_generate.return_value = img
    gen = FluxKleinGenerator()
    out = gen.generate_image(prompt="a cat")
    assert out.size == (1024, 1024)
    assert out.mode == "RGB"
    mock_generate.assert_called_once()


def test_triposplat_generator_interface(tmp_path):
    # We only verify the constructor signature and method existence here.
    # Real model loading requires downloaded checkpoints.
    assert hasattr(TripoSplatGenerator, "image_to_splat")
    assert hasattr(TripoSplatGenerator, "preprocess_image")


def test_pipeline_run_mocked(tmp_path):
    cfg = PipelineConfig(triposplat_dir=tmp_path)
    with patch("tripoflux.pipeline.FluxKleinGenerator"), \
         patch("tripoflux.pipeline.BiRefNetMLX"), \
         patch("tripoflux.pipeline.TripoSplatGenerator"):
        pipe = TripoFluxPipeline(cfg)

    fake_rgb = Image.new("RGB", (1024, 1024), color="blue")
    fake_rgba = Image.new("RGBA", (1024, 1024), color=(0, 255, 0, 255))
    fake_prepared = Image.new("RGB", (1024, 1024), color="black")
    fake_ply = b"fake-ply"
    fake_splat = b"fake-splat"
    fake_spz = b"fake-spz"

    with patch.object(pipe, "generate_image", return_value=fake_rgb) as m_gen, \
         patch.object(pipe, "remove_background", return_value=fake_rgba) as m_rm, \
         patch.object(pipe, "generate_splat", return_value=(fake_ply, fake_splat, fake_spz, fake_prepared)) as m_splat:
        result = pipe.run("a test prompt")

    assert result.generated_image is fake_rgb
    assert result.rgba_image is fake_rgba
    assert result.ply_bytes == fake_ply
    assert result.splat_bytes == fake_splat
    assert result.spz_bytes == fake_spz
    assert result.prompt == "a test prompt"
    m_gen.assert_called_once()
    m_rm.assert_called_once()
    m_splat.assert_called_once()


def test_birefnet_mlx_placeholder(tmp_path):
    from tripoflux.models.birefnet_mlx import BiRefNetMLX

    with patch("tripoflux.models.birefnet_mlx.BiRefNetCoreML"):
        remover = BiRefNetMLX(triposplat_dir=tmp_path)
    assert remover.backend_name.startswith("mlx")
