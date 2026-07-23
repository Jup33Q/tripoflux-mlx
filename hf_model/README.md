---
license: other
license_name: mixed-see-license-section
tags:
- mlx
- triposplat
- 3d-gaussian-splatting
- image-to-3d
- apple-silicon
---

# TripoSplat MLX

MLX-native implementation of [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat) for Apple Silicon.

This repository contains MLX ports of the TripoSplat submodules:

- **`mlx_models/dinov3_mlx.py`** — DinoV3 ViT-H/16+ image encoder
- **`mlx_models/flux2vae_mlx.py`** — Flux2 VAE encoder
- **`mlx_models/flow_mlx.py`** — LatentSeqMMFlowModel (24-block multimodal flow transformer)

The original TripoSplat decoder (`OctreeGaussianDecoder`) remains on PyTorch MPS due to its complex dynamic sampling logic.

## Weights

| File | Size | Description |
|------|------|-------------|
| `dino_v3_vit_h.safetensors` | 1.6 GB | DinoV3 ViT-H/16+ image encoder |
| `flux2-vae.safetensors` | 336 MB | Flux2 VAE encoder |
| `triposplat_fp16.safetensors` | 741 MB | LatentSeqMMFlowModel (flow transformer) |
| `triposplat_vae_decoder_fp16.safetensors` | 576 MB | TripoSplat VAE decoder |
| `birefnet.safetensors` | 444 MB | BiRefNet background removal |

## License

This is a **mixed-license repository**. The most restrictive artifact determines
what the repository as a whole can be used for — with `flux2-vae.safetensors`
included, the repo as a whole is **non-commercial only**.

### ⛔ Non-commercial — separate restriction

| File | Upstream | License | Terms |
|------|----------|---------|-------|
| `flux2-vae.safetensors` | [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) | **FLUX Non-Commercial License** | Weights are gated upstream. Commercial use of the weights or their outputs requires a paid license from Black Forest Labs. If you need a fully permissive stack, delete this file and load the VAE from the upstream gated repo at runtime after accepting its terms. |

### ⚠️ Custom permissive license — attribution required on redistribution

| File | Upstream | License | Terms |
|------|----------|---------|-------|
| `dino_v3_vit_h.safetensors` | [facebook/dinov3](https://github.com/facebookresearch/dinov3) | **DINOv3 License** | Commercial use **is allowed**, but any redistribution of copies or derivatives must include the DINOv3 License text. Upstream downloads are gated and unavailable in sanctioned jurisdictions. |

### ✅ MIT — permissive core

| File | Upstream | License |
|------|----------|---------|
| `triposplat_fp16.safetensors` | [VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat) | MIT |
| `triposplat_vae_decoder_fp16.safetensors` | [VAST-AI/TripoSplat](https://huggingface.co/VAST-AI/TripoSplat) | MIT |
| `birefnet.safetensors` | [ZhengPeng7/BiRefNet](https://huggingface.co/ZhengPeng7/BiRefNet) | MIT |
| `mlx_models/*.py` | this repo (MLX port) | MIT |

The MLX port code in `mlx_models/` is released under the
[MIT License](https://opensource.org/licenses/MIT), same as the main project:
[https://github.com/Jup33Q/tripoflux-mlx](https://github.com/Jup33Q/tripoflux-mlx).

## Usage

```python
from mlx_models.dinov3_mlx import DinoV3ViT
from mlx_models.flux2vae_mlx import Flux2VAEEncoder
from mlx_models.flow_mlx import LatentSeqMMFlowModel

# Load models
dino = DinoV3ViT()
dino.load_safetensors("dino_v3_vit_h.safetensors")

vae = Flux2VAEEncoder()
vae.load_safetensors("flux2-vae.safetensors")

flow = LatentSeqMMFlowModel(
    q_token_length=8192, in_channels=16, cam_channels=5, out_channels=16,
    model_channels=1024, cond_channels=1280, cond2_channels=128,
    num_refiner_blocks=2, num_blocks=24, num_heads=16, mlp_ratio=4,
    qk_rms_norm=True, share_mod=True, use_shift_table=True,
)
flow.load_safetensors("triposplat_fp16.safetensors")
```

## Full Pipeline

For the complete text-to-3D pipeline (FLUX.2-klein-9B → BiRefNet → TripoSplat), see the main project:
[https://github.com/Jup33Q/tripoflux-mlx](https://github.com/Jup33Q/tripoflux-mlx)
