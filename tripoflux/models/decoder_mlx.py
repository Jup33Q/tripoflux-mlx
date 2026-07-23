"""MLX port of the TripoSplat OctreeGaussianDecoder.

Status: **partial / experimental**

The full decoder contains two parts:
1. `OctreeProbabilityFixedlenDecoder` — octree probability decoding with
   dynamic per-level sampling. The `sample()` method uses dynamic shapes,
   recursive subdivision, and a custom systematic sampling algorithm that
   are difficult to express efficiently in MLX's static graph model.
2. `ElasticGaussianFixedlenDecoder` — transformer-based Gaussian parameter
   decoding. This part is mostly static and *could* be ported, but it
   depends on the output of the octree sampler.

Current strategy:
- Keep the octree probability sampler on PyTorch MPS.
- Keep the elastic Gaussian decoder on PyTorch MPS.
- Reuse the already-ported MLX encoders and flow model for the heavy
  lifting upstream.

If a full MLX decoder is needed in the future, the recommended path is:
1. Port `ElasticGaussianFixedlenDecoder` first (static transformer).
2. Replace `sample_probs` with a Gumbel-max or categorical sampler that
   MLX supports natively.
3. Accept a small numerical difference in the sampled points, or use a
   hybrid approach where MLX does the forward pass and PyTorch does the
   sampling.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np
import safetensors.torch
import torch

from .flow_mlx import (
    FeedForwardNet,
    LayerNorm32,
    MultiHeadRMSNorm,
    PcdAbsolutePositionEmbedderV2,
    RopeMultiHeadAttention,
    TimestepEmbedder,
    _layer_norm,
)

logger = logging.getLogger(__name__)


class TransformerCrossBlock(nn.Module):
    """Cross-attention transformer block (PyTorch parity)."""

    def __init__(self, channels: int, ctx_channels: int, num_heads: int,
                 mlp_ratio: float = 4.0, qk_rms_norm: bool = True,
                 qk_rms_norm_cross: bool = True, qkv_bias: bool = True):
        super().__init__()
        self.norm1 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=True, eps=1e-6)
        self.norm3 = LayerNorm32(channels, elementwise_affine=False, eps=1e-6)
        self.self_attn = RopeMultiHeadAttention(
            channels, num_heads=num_heads, type="self",
            qkv_bias=qkv_bias, qk_rms_norm=qk_rms_norm, use_rope=False,
        )
        self.cross_attn = RopeMultiHeadAttention(
            channels, ctx_channels=ctx_channels, num_heads=num_heads,
            type="cross", qkv_bias=qkv_bias, qk_rms_norm=qk_rms_norm_cross,
            use_rope=False,
        )
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)

    def __call__(self, x: mx.array, context: mx.array) -> mx.array:
        x = x + self.self_attn(self.norm1(x))
        x = x + self.cross_attn(self.norm2(x), context)
        x = x + self.mlp(self.norm3(x))
        return x


class ElasticGaussianFixedlenDecoderMLX(nn.Module):
    """MLX port of the ElasticGaussianFixedlenDecoder (experimental).

    This is a best-effort port of the static transformer part. The dynamic
    octree sampling that feeds into this decoder is still on PyTorch MPS.
    """

    def __init__(self, in_channels: int, model_channels: int, cond_channels: int,
                 num_blocks: int, num_heads: int = 16, num_head_channels: int = 64,
                 mlp_ratio: float = 4.0, qk_rms_norm: bool = True,
                 qk_rms_norm_cross: bool = True):
        super().__init__()
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.num_blocks = num_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio

        self.input_layer = nn.Linear(in_channels, model_channels)
        self.in_proj = nn.Linear(in_channels, model_channels)
        self.pos_embedder = PcdAbsolutePositionEmbedderV2(channels=model_channels, in_channels=3)

        self.blocks = [
            TransformerCrossBlock(
                model_channels, ctx_channels=cond_channels,
                num_heads=self.num_heads, mlp_ratio=self.mlp_ratio,
                qk_rms_norm=qk_rms_norm, qk_rms_norm_cross=qk_rms_norm_cross,
            )
            for _ in range(num_blocks)
        ]

    def __call__(self, x: dict, cond: mx.array) -> mx.array:
        pcd = x["points"]
        d = self.input_layer.weight.dtype
        B, L, C = pcd.shape
        h = self.in_proj(pcd.astype(d)) + self.pos_embedder(pcd.reshape(-1, 3)).reshape(B, L, -1).astype(d)
        for block in self.blocks:
            h = block(h, cond)
        return h

    def load_safetensors(self, path: str) -> None:
        state_dict = safetensors.torch.load_file(path)
        our_sd = dict(mlx.utils.tree_flatten(self.parameters()))
        loaded = {}
        for k, v in state_dict.items():
            if k in our_sd:
                if v.shape != our_sd[k].shape:
                    raise ValueError(f"Shape mismatch {k}: {v.shape} vs {our_sd[k].shape}")
                if v.dtype == torch.bfloat16:
                    v = v.to(torch.float16)
                loaded[k] = mx.array(v.numpy())
        missing = set(our_sd) - set(loaded)
        unexpected = set(loaded) - set(our_sd)
        if missing:
            raise KeyError(f"[Decoder-MLX] Missing keys: {missing}")
        if unexpected:
            raise KeyError(f"[Decoder-MLX] Unexpected keys: {unexpected}")
        self.load_weights(list(loaded.items()), strict=False)


def load_elastic_decoder_mlx(path: str, **kwargs) -> ElasticGaussianFixedlenDecoderMLX:
    model = ElasticGaussianFixedlenDecoderMLX(**kwargs)
    model.load_safetensors(path)
    return model
