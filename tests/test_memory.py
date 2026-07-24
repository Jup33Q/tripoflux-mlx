"""Tests for the memory hygiene mechanisms."""

import tripoflux.server as server
from tripoflux.mem_utils import release_gpu_caches
from tripoflux.models.flux_klein_mlx import FluxKleinGenerator


def test_release_gpu_caches_no_raise():
    release_gpu_caches()


def test_flux_unload_methods():
    g = FluxKleinGenerator(backend="mlx", quantize=8)
    g._mflux_model = object()
    g._diffusers_pipe = object()
    g.unload_diffusers()
    assert g._diffusers_pipe is None
    assert g._mflux_model is not None  # diffusers unload must not touch mflux
    g.unload()
    assert g._mflux_model is None


def test_evict_old_jobs(monkeypatch):
    monkeypatch.setattr(server, "_MAX_JOBS", 3)
    server._jobs.clear()
    try:
        for i in range(5):
            server._jobs[f"job{i}"] = {"status": "completed", "result": b"x" * 10}
        server._jobs["running"] = {"status": "running", "result": None}
        server._evict_old_jobs()
        # 6 jobs with cap 3 → the 3 oldest finished jobs are evicted;
        # the running job is never touched.
        assert "running" in server._jobs
        remaining = [j for j in server._jobs if j.startswith("job")]
        assert remaining == ["job3", "job4"]
    finally:
        server._jobs.clear()


def test_evict_old_jobs_under_cap_keeps_everything():
    server._jobs.clear()
    try:
        for i in range(server._MAX_JOBS):
            server._jobs[f"job{i}"] = {"status": "completed", "result": b"x"}
        server._evict_old_jobs()
        assert len(server._jobs) == server._MAX_JOBS
    finally:
        server._jobs.clear()
