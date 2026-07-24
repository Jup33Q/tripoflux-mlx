# TripoFlux MLX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform: macOS Apple Silicon](https://img.shields.io/badge/platform-macOS%20Apple%20Silicon-lightgrey)](https://support.apple.com/en-us/HT211814)
[![MLX](https://img.shields.io/badge/MLX-native-green)](https://github.com/ml-explore/mlx)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Models-blue)](https://huggingface.co/Jup33QE/tripoflux-mlx)
[![ModelScope](https://img.shields.io/badge/ModelScope-Models-purple)](https://modelscope.cn/models/Jup33Q/TripoSplat-MLX)
[![GitHub stars](https://img.shields.io/github/stars/Jup33Q/tripoflux-mlx?style=social)](https://github.com/Jup33Q/tripoflux-mlx)
[![GitHub issues](https://img.shields.io/github/issues/Jup33Q/tripoflux-mlx)](https://github.com/Jup33Q/tripoflux-mlx/issues)

MLX / CoreML accelerated pipeline for **FLUX.2-klein-9B → BiRefNet → TripoSplat** on Apple Silicon.

## Demo

[![LKG WebXR holographic preview demo](docs/demo-cover.png)](https://www.bilibili.com/video/BV1bFge6qEK2)

📺 [Bilibili video: Looking Glass (LKG) WebXR holographic preview demo](https://www.bilibili.com/video/BV1bFge6qEK2)

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
- **Background removal** via SAM3 / DA2 / BiRefNet, preferring MLX, falling back to CoreML or PyTorch MPS. SAM3 segments by the generation prompt's subject (open-vocabulary), so large subjects like buildings and trees are kept instead of being saliency-cut as background. Hub-backed removers fail fast to local caches when offline.
- **3D Gaussian Splatting** via the official [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) model.
- **R3F web UI**: React Three Fiber splat viewport with OrbitControls, inline generation progress, log terminal, and **Looking Glass (LKG) holographic preview** via `@lookingglass/webxr` (WASD orbits, arrow keys dolly).
- **Smooth model transitions**: the viewport lerp/slerp-morphs from the current model to each newly generated splat.
- **MLX ports** for TripoSplat submodules:
  - ✅ `DinoV3ViT` image encoder (`tripoflux/models/dinov3_mlx.py`)
  - ✅ `Flux2VAEEncoder` (`tripoflux/models/flux2vae_mlx.py`)
  - ✅ `LatentSeqMMFlowModel` flow transformer (`tripoflux/models/flow_mlx.py`)
  - ⏳ `OctreeGaussianDecoder` (remains on PyTorch MPS; complex dynamic sampling)
- **DA2 CoreML** models for fast depth-based background removal (base + large).
- **Memory hygiene**: all jobs run on a single worker thread (MLX streams are thread-local), GPU allocator caches are released between jobs, temporary quantize variants of FLUX are disposed immediately after use, the ~34 GB diffusers fallback is unloaded once mflux is healthy again, and old job results are LRU-evicted (keep last 8).

## TODO

### Completed
- [x] FLUX.2-klein-9B MLX inference via mflux
- [x] BiRefNet background removal (MLX/CoreML/MPS fallback)
- [x] TripoSplat MLX ports: DinoV3ViT, Flux2VAEEncoder, LatentSeqMMFlowModel
- [x] Hybrid MLX/MPS TripoSplat pipeline
- [x] FastAPI backend with SSE progress streaming
- [x] R3F web UI: splat viewer + OrbitControls, inline progress, terminal
- [x] Looking Glass WebXR preview (LKG quilt) with keyboard orbit/dolly
- [x] Splat-to-splat lerp/slerp morph transitions
- [x] SPZ as the canonical splat artifact — viewport preview, `.ply` and `.splat` exports are all derived from it (WYSIWYG with the SPZ download)
- [x] SAM3 background removal via mlx-vlm
- [x] DA2 CoreML background removal (base + large)
- [x] mflux PR #481: Support mlx 0.32.x

### In Progress
- [ ] OctreeGaussianDecoder full MLX port (dynamic sampling logic)
- [ ] BiRefNet native MLX port (Swin-L + deformable ASPP)
- [ ] mlx-community submission for TripoSplat MLX weights

### Planned
- [ ] CoreML conversion for static encoders (DinoV3, VAE)
- [ ] Batch generation API
- [ ] macOS Swift app deployment (MLX Swift)
- [ ] 4-bit quantized TripoSplat weights

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
#    We currently use our own fork, which carries PR #481 (mlx 0.32.x support)
#    on top of upstream 0.18.0. This is already pinned in pyproject.toml, so
#    `pip install -e .` above pulls it in automatically; to (re)install it
#    standalone:
pip install "mflux @ git+https://github.com/Jup33Q/mflux.git@main"

# 4. Download model weights manually
#    a) TripoSplat checkpoints → ckpts/VAST-AI/TripoSplat/
#       - diffusion_models/triposplat_fp16.safetensors
#       - clip_vision/dino_v3_vit_h.safetensors
#       - vae/triposplat_vae_decoder_fp16.safetensors
#       - vae/flux2-vae.safetensors
#       - background_removal/birefnet.safetensors
#       (also mirrored on ModelScope: https://modelscope.cn/models/Jup33Q/TripoSplat-MLX)
#    b) FLUX.2-klein-9B is fetched automatically by mflux on first use.
#    c) (Optional) DA2 CoreML models from HuggingFace:
#       - da2_base.mlpackage
#       - da2_large.mlpackage

# 5. Install frontend dependencies
cd frontend && npm install && cd ..

# 6. Start the web UI (production: build the frontend, then serve via FastAPI)
cd frontend && npm run build && cd ..
python -m tripoflux.server
# open http://localhost:8321
```

### Development mode

Run the Vite dev server (hot reload, proxies `/api` to the backend on :8321)
alongside the FastAPI backend:

```bash
python -m tripoflux.server          # terminal 1 → http://127.0.0.1:8321
cd frontend && npm run dev          # terminal 2 → http://localhost:5173
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
├── frontend/                     # web UI (Vite + React 19 + TypeScript + React Three Fiber)
├── scripts/
│   ├── download_models.sh        # fetch all required weights
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
