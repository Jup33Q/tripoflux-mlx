"""MLX port of the DinoV3 ViT-H/16+ image encoder used by TripoSplat.

This is a line-by-line port of the PyTorch reference in
`tripoflux/vendor/triposplat/model.py`. Weights are loaded from the same
safetensors checkpoint and converted to MLX arrays.
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np
import safetensors.torch
import torch


def _rotate_half(x: mx.array) -> mx.array:
    x1, x2 = mx.split(x, 2, axis=-1)
    return mx.concatenate([-x2, x1], axis=-1)


class DinoV3PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 1280):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, C, H, W) NCHW -> MLX Conv2d expects NHWC
        x = x.transpose(0, 2, 3, 1)
        x = self.proj(x)
        # -> (B, H/P, W/P, embed_dim) -> (B, N, embed_dim)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        return x


class DinoV3RotaryEmbedding2D(nn.Module):
    def __init__(self, dim: int, base: float = 100.0):
        super().__init__()
        self.inv_freq = 1.0 / (base ** mx.arange(0, 1, 4.0 / dim, dtype=mx.float32))

    def __call__(self, height: int, width: int, dtype=mx.float32) -> Tuple[mx.array, mx.array]:
        coords_h = mx.arange(0.5, height, dtype=mx.float32) / height
        coords_w = mx.arange(0.5, width, dtype=mx.float32) / width
        coords = mx.stack(mx.meshgrid(coords_h, coords_w, indexing="ij"), axis=-1)
        coords = (2.0 * coords - 1.0).flatten(0, 1)
        angles = mx.tile((2 * math.pi * coords[:, :, None] * self.inv_freq[None, None, :]).flatten(1, 2), 2)
        cos = angles.cos()[None, None, :, :]
        sin = angles.sin()[None, None, :, :]
        return cos.astype(dtype), sin.astype(dtype)


class DinoV3Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: tuple = (True, False, True)):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        q_bias, k_bias, v_bias = qkv_bias
        self.q_proj = nn.Linear(dim, dim, bias=q_bias)
        self.k_proj = nn.Linear(dim, dim, bias=k_bias)
        self.v_proj = nn.Linear(dim, dim, bias=v_bias)
        self.o_proj = nn.Linear(dim, dim, bias=True)

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, num_prefix_tokens: int = 0) -> mx.array:
        B, N, C = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, N, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        if num_prefix_tokens > 0:
            q_pre, q_pat = mx.split(q, [num_prefix_tokens], axis=-2)
            k_pre, k_pat = mx.split(k, [num_prefix_tokens], axis=-2)
            q = mx.concatenate([q_pre, q_pat * cos + _rotate_half(q_pat) * sin], axis=-2)
            k = mx.concatenate([k_pre, k_pat * cos + _rotate_half(k_pat) * sin], axis=-2)
        else:
            q = q * cos + _rotate_half(q) * sin
            k = k * cos + _rotate_half(k) * sin

        scale = 1.0 / math.sqrt(self.head_dim)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, N, C))


class DinoV3MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, bias: bool = True):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=bias)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class DinoV3Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 qkv_bias: tuple = (True, False, True), layerscale_init: float = 1.0,
                 mlp_bias: bool = True, eps: float = 1e-5):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=eps)
        self.attn = DinoV3Attention(dim, num_heads, qkv_bias=qkv_bias)
        self.ls1 = mx.ones(dim) * layerscale_init
        self.norm2 = nn.LayerNorm(dim, eps=eps)
        self.mlp = DinoV3MLP(dim, int(dim * mlp_ratio), bias=mlp_bias)
        self.ls2 = mx.ones(dim) * layerscale_init

    def __call__(self, x: mx.array, cos: mx.array, sin: mx.array, num_prefix_tokens: int = 0) -> mx.array:
        x = x + self.ls1 * self.attn(self.norm1(x), cos, sin, num_prefix_tokens=num_prefix_tokens)
        x = x + self.ls2 * self.mlp(self.norm2(x))
        return x


class DinoV3ViT(nn.Module):
    def __init__(self, hidden_size: int = 1280, num_heads: int = 20, num_layers: int = 32,
                 patch_size: int = 16, num_register_tokens: int = 4,
                 intermediate_size: int = 5120, layerscale_init: float = 1.0,
                 query_bias: bool = True, key_bias: bool = False, value_bias: bool = True,
                 mlp_bias: bool = True, rope_theta: float = 100.0, layer_norm_eps: float = 1e-5):
        super().__init__()
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.patch_embed = DinoV3PatchEmbed(patch_size=patch_size, embed_dim=hidden_size)
        self.cls_token = mx.zeros((1, 1, hidden_size))
        self.register_tokens = mx.zeros((1, num_register_tokens, hidden_size))
        self.rope = DinoV3RotaryEmbedding2D(dim=hidden_size // num_heads, base=rope_theta)
        qkv_bias = (query_bias, key_bias, value_bias)
        self.blocks = [
            DinoV3Block(hidden_size, num_heads, mlp_ratio=intermediate_size / hidden_size,
                        qkv_bias=qkv_bias, layerscale_init=layerscale_init,
                        mlp_bias=mlp_bias, eps=layer_norm_eps)
            for _ in range(num_layers)
        ]
        self.norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        B, _, H, W = pixel_values.shape
        x = self.patch_embed(pixel_values)
        hp, wp = H // self.patch_size, W // self.patch_size
        cos, sin = self.rope(hp, wp, dtype=x.dtype)
        x = mx.concatenate([mx.tile(self.cls_token, (B, 1, 1)),
                            mx.tile(self.register_tokens, (B, 1, 1)), x], axis=1)
        num_prefix = 1 + self.num_register_tokens
        for block in self.blocks:
            x = block(x, cos, sin, num_prefix_tokens=num_prefix)
        return self.norm(x)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------
    def load_safetensors(self, path: str) -> None:
        state_dict = safetensors.torch.load_file(path)
        our_sd = dict(mlx.utils.tree_flatten(self.parameters()))
        loaded = {}
        for hf_key in state_dict:
            k = (hf_key
                 .replace("embeddings.patch_embeddings.", "patch_embed.proj.")
                 .replace("embeddings.cls_token", "cls_token")
                 .replace("embeddings.mask_token", "mask_token")
                 .replace("embeddings.register_tokens", "register_tokens"))
            m = re.match(r"layer\.(\d+)\.(.+)", k)
            if m:
                rest = m.group(2)
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    rest = rest.replace(f"attention.{proj}", f"attn.{proj}")
                rest = (rest.replace("layer_scale1.lambda1", "ls1")
                            .replace("layer_scale2.lambda1", "ls2"))
                k = f"blocks.{m.group(1)}.{rest}"
            if k in our_sd:
                tensor = state_dict[hf_key]
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float16)
                # MLX Conv2d expects NHWC weight layout (out, kh, kw, in)
                if tensor.ndim == 4 and tensor.shape[1] < tensor.shape[0]:
                    tensor = tensor.permute(0, 2, 3, 1)
                assert tensor.shape == our_sd[k].shape, \
                    f"Shape mismatch {k}: {tensor.shape} vs {our_sd[k].shape}"
                loaded[k] = mx.array(tensor.numpy())
        check_sd = {k: v for k, v in our_sd.items() if k not in ("mask_token", "rope.inv_freq")}
        missing = set(check_sd) - set(loaded)
        unexpected = set(loaded) - set(check_sd)
        if missing:
            raise KeyError(f"[DINOv3-MLX] Missing keys: {missing}")
        if unexpected:
            raise KeyError(f"[DINOv3-MLX] Unexpected keys: {unexpected}")
        self.load_weights(list(loaded.items()), strict=False)


def load_dinov3_mlx(path: str) -> DinoV3ViT:
    model = DinoV3ViT()
    model.load_safetensors(path)
    return model
