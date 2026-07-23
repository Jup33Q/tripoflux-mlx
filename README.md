# TripoFlux MLX

MLX / CoreML accelerated pipeline for **FLUX.2-klein-9B → BiRefNet → TripoSplat** on Apple Silicon.

```text
prompt ──► FLUX.2-klein-9B (MLX) ──► RGB image
                                        │
                                        ▼
                              BiRefNet (MLX / CoreML / MPS)
                                        │
                                        ▼
                              RGBA cutout
                                        │
                                        ▼
                              TripoSplat (MPS, optional MLX)
                                        │
                                        ▼
                         Gaussian Splat (.ply / .splat)
```

## Features

- **MLX-first** image generation with `mflux` / `FLUX.2-klein-9B`.
- **Background removal** via BiRefNet, preferring MLX, falling back to CoreML or PyTorch MPS.
- **3D Gaussian Splatting** via the official [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) model.
- Web UI with image preview and `.splat` viewer.
- **MLX ports** for TripoSplat submodules:
  - ✅ `DinoV3ViT` image encoder (`tripoflux/models/dinov3_mlx.py`)
  - ✅ `Flux2VAEEncoder` (`tripoflux/models/flux2vae_mlx.py`)
  - ✅ `LatentSeqMMFlowModel` flow transformer (`tripoflux/models/flow_mlx.py`)
  - ⏳ `OctreeGaussianDecoder` (remains on PyTorch MPS; complex dynamic sampling)

## Requirements

- macOS 14+ on Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- ~40 GB free disk space for model weights

## Quick start

```bash
# 1. Clone / enter the project
cd tripoflux-mlx

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Install mflux (native MLX FLUX.2 runtime)
pip install mflux -i https://pypi.tuna.tsinghua.edu.cn/simple
# If the default PyPI index is slow, the mirror above is recommended.

# 4. Download model weights manually
#    a) TripoSplat checkpoints → ckpts/VAST-AI/TripoSplat/
#       - diffusion_models/triposplat_fp16.safetensors
#       - clip_vision/dino_v3_vit_h.safetensors
#       - vae/triposplat_vae_decoder_fp16.safetensors
#       - vae/flux2-vae.safetensors
#       - background_removal/birefnet.safetensors
#    b) FLUX.2-klein-9B is fetched automatically by mflux on first use.

# 5. Start the web UI
python -m tripoflux.server
# open http://localhost:8000
```

## Quantization

FLUX.2-klein-9B supports 4-bit and 8-bit quantization via mflux, reducing memory and disk usage at the cost of a small quality drop.

- **None**: full precision (largest, ~34 GB)
- **8-bit** (default): balanced quality and size
- **4-bit**: smallest and fastest, slightly lower fidelity

You can set the default in `configs/default.yaml`:

```yaml
quantization:
  flux_bits: 8   # 4 | 8 | null
```

Or choose per-request in the web UI dropdown.

## Project layout

```
tripoflux-mlx/
├── configs/default.yaml          # model paths & default parameters
├── frontend/                     # web UI (HTML/JS)
├── scripts/
│   ├── download_models.py        # fetch all required weights
│   └── convert_birefnet_coreml.py
├── tests/
│   └── test_pipeline.py
└── tripoflux/
    ├── models/
    │   ├── flux_klein_mlx.py         # FLUX.2-klein-9B MLX inference (mflux)
    │   ├── birefnet_mlx.py           # BiRefNet MLX-first wrapper
    │   ├── birefnet_coreml.py        # BiRefNet CoreML fallback
    │   ├── dinov3_mlx.py             # MLX DinoV3ViT
    │   ├── flux2vae_mlx.py           # MLX Flux2VAEEncoder
    │   ├── flow_mlx.py               # MLX LatentSeqMMFlowModel
    │   ├── triposplat_mlx_pipeline.py# Hybrid MLX/MPS TripoSplat pipeline
    │   ├── triposplat_wrapped.py     # TripoSplat wrapper (MPS/MLX)
    │   └── coreml_utils.py           # CoreML conversion helpers
    ├── pipeline.py                   # end-to-end pipeline
    └── server.py                     # FastAPI app
```

## License

MIT. TripoSplat model weights and code are MIT-licensed by VAST-AI-Research. FLUX.2-klein-9B is released under the FLUX Non-Commercial License.
