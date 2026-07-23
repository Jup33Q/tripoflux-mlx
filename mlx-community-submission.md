# Submission: TripoSplat MLX for mlx-community

## Project Overview

**TripoSplat MLX** is a native Apple Silicon (MLX) implementation of [TripoSplat](https://github.com/VAST-AI-Research/TripoSplat), a single-image-to-3D-Gaussian-Splat model by VAST-AI-Research.

This project ports the core TripoSplat submodules from PyTorch to MLX, enabling efficient on-device inference for 3D content creation on Mac.

## What Has Been Done

### MLX Ports (Complete)

| Component | Original | MLX Port | Status |
|-----------|----------|----------|--------|
| DinoV3 ViT-H/16+ | PyTorch | `mlx_models/dinov3_mlx.py` | ✅ Working |
| Flux2 VAE Encoder | PyTorch | `mlx_models/flux2vae_mlx.py` | ✅ Working |
| LatentSeqMMFlowModel | PyTorch | `mlx_models/flow_mlx.py` | ✅ Working |
| OctreeGaussianDecoder | PyTorch | — | ⏳ Remains on MPS (complex dynamic sampling) |

### Key Features

- **Native MLX**: No PyTorch dependency for encoders and flow model
- **Weight-compatible**: Loads original safetensors checkpoints directly
- **Numerically verified**: Outputs match PyTorch reference within tolerance
- **End-to-end pipeline**: Includes FLUX.2-klein-9B → BiRefNet → TripoSplat → SPZ export
- **MIT License**: Fully open source

## Links

- **HuggingFace Model**: https://huggingface.co/Jup33QE/tripoflux-mlx
- **GitHub Project**: https://github.com/Jup33Q/tripoflux-mlx
- **Original TripoSplat**: https://github.com/VAST-AI-Research/TripoSplat

## Request

We would like to contribute this MLX implementation to the **mlx-community** organization on HuggingFace. The model weights and MLX code are ready for transfer.

If the maintainers are interested, we can:

1. Transfer the HuggingFace repository to `mlx-community/triposplat-mlx`
2. Submit a PR to the mlx-explore/mlx-examples repository with the model implementations
3. Provide any additional verification or benchmarks needed

## Contact

- **Author**: Jup33Q (zxgzg@163.com)
- **HuggingFace**: https://huggingface.co/Jup33QE
- **GitHub**: https://github.com/Jup33Q

---

*Submitted: 2026-07-23*
