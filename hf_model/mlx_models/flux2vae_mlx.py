"""MLX port of the Flux2 VAE encoder used by TripoSplat."""

from __future__ import annotations

import math
import re
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np
import safetensors.torch
import torch


class Flux2ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, use_shortcut: bool = False):
        super().__init__()
        self.norm1 = nn.GroupNorm(32, in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(32, out_channels, eps=1e-6)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0) if use_shortcut else None

    def __call__(self, x: mx.array) -> mx.array:
        # x: NHWC
        h = nn.silu(self.norm1(x))
        h = nn.silu(self.norm2(self.conv1(h)))
        h = self.conv2(h)
        return h + (self.conv_shortcut(x) if self.conv_shortcut is not None else x)


class Flux2Downsampler(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=0)

    def __call__(self, x: mx.array) -> mx.array:
        # x: NHWC; pad (0,1,0,1) in NCHW corresponds to (0,0,0,1,0,1) in NHWC
        x = mx.pad(x, [(0, 0), (0, 1), (0, 1), (0, 0)])
        return self.conv(x)


class Flux2Attention(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.group_norm = nn.GroupNorm(32, channels, eps=1e-6)
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.to_out = nn.Linear(channels, channels)

    def __call__(self, x: mx.array) -> mx.array:
        # x: NHWC
        B, H, W, C = x.shape
        h = self.group_norm(x).reshape(B, H * W, C)
        q = self.to_q(h).reshape(B, -1, 1, C).transpose(0, 2, 1, 3)
        k = self.to_k(h).reshape(B, -1, 1, C).transpose(0, 2, 1, 3)
        v = self.to_v(h).reshape(B, -1, 1, C).transpose(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(C)
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
        out = self.to_out(out.transpose(0, 2, 1, 3).reshape(B, -1, C))
        return x + out.reshape(B, H, W, C)


class Flux2Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_in = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1)
        self.down_0_resnets = [Flux2ResnetBlock(128, 128), Flux2ResnetBlock(128, 128)]
        self.down_0_sampler = Flux2Downsampler(128)
        self.down_1_resnets = [Flux2ResnetBlock(128, 256, use_shortcut=True), Flux2ResnetBlock(256, 256)]
        self.down_1_sampler = Flux2Downsampler(256)
        self.down_2_resnets = [Flux2ResnetBlock(256, 512, use_shortcut=True), Flux2ResnetBlock(512, 512)]
        self.down_2_sampler = Flux2Downsampler(512)
        self.down_3_resnets = [Flux2ResnetBlock(512, 512), Flux2ResnetBlock(512, 512)]
        self.mid_attn = Flux2Attention(512)
        self.mid_resnets = [Flux2ResnetBlock(512, 512), Flux2ResnetBlock(512, 512)]
        self.conv_norm_out = nn.GroupNorm(32, 512, eps=1e-6)
        self.conv_out = nn.Conv2d(512, 64, kernel_size=3, stride=1, padding=1)

    def __call__(self, x: mx.array) -> mx.array:
        # x: NCHW -> NHWC for MLX conv
        x = x.transpose(0, 2, 3, 1)
        x = self.conv_in(x)
        for r in self.down_0_resnets: x = r(x)
        x = self.down_0_sampler(x)
        for r in self.down_1_resnets: x = r(x)
        x = self.down_1_sampler(x)
        for r in self.down_2_resnets: x = r(x)
        x = self.down_2_sampler(x)
        for r in self.down_3_resnets: x = r(x)
        x = self.mid_resnets[0](x)
        x = self.mid_attn(x)
        x = self.mid_resnets[1](x)
        x = self.conv_out(nn.silu(self.conv_norm_out(x)))
        # NHWC -> NCHW
        return x.transpose(0, 3, 1, 2)


class Flux2VAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Flux2Encoder()
        self.quant_conv = nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.bn = nn.BatchNorm(128, eps=1e-5, momentum=0.1, affine=False, track_running_stats=True)

    def encode(self, images: mx.array, deterministic: bool = True, seed: Optional[int] = None) -> mx.array:
        # images: NCHW
        moments = self.quant_conv(self.encoder(images).transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
        mean, logvar = mx.split(moments, 2, axis=1)
        if deterministic:
            latents = mean
        else:
            if seed is not None:
                mx.random.seed(seed)
            noise = mx.random.normal(mean.shape, dtype=mean.dtype)
            latents = mean + mx.exp(0.5 * logvar) * noise
        B, C, H, W = latents.shape
        latents = latents.reshape(B, C, H // 2, 2, W // 2, 2).transpose(0, 1, 3, 5, 2, 4)
        latents = latents.reshape(B, C * 4, H // 2, W // 2)
        # BatchNorm over channel dim (dim=1)
        latents_nhwc = latents.transpose(0, 2, 3, 1)
        latents_nhwc = self.bn(latents_nhwc)
        latents = latents_nhwc.transpose(0, 3, 1, 2)
        return latents.astype(mx.float32).flatten(2).transpose(0, 2, 1)

    def load_safetensors(self, path: str) -> None:
        state_dict = safetensors.torch.load_file(path)
        our_sd = dict(mlx.utils.tree_flatten(self.parameters()))
        loaded = {}
        for k, v in state_dict.items():
            if k.startswith(("decoder.", "post_quant_conv.")):
                continue
            m = re.match(r"encoder\.down_blocks\.(\d+)\.resnets\.(\d+)\.(.+)", k)
            if m:
                k = f"encoder.down_{m.group(1)}_resnets.{m.group(2)}.{m.group(3)}"
            else:
                m = re.match(r"encoder\.down_blocks\.(\d+)\.downsamplers\.0\.(.+)", k)
                if m:
                    k = f"encoder.down_{m.group(1)}_sampler.{m.group(2)}"
                else:
                    m = re.match(r"encoder\.mid_block\.resnets\.(\d+)\.(.+)", k)
                    if m:
                        k = f"encoder.mid_resnets.{m.group(1)}.{m.group(2)}"
                    else:
                        m = re.match(r"encoder\.mid_block\.attentions\.0\.(.+)", k)
                        if m:
                            k = f"encoder.mid_attn.{m.group(1)}"
                            # PyTorch ModuleList to_out.0.weight -> MLX Linear to_out.weight
                            k = k.replace("to_out.0.", "to_out.")
            if k in our_sd:
                if v.dtype == torch.bfloat16:
                    v = v.to(torch.float16)
                # MLX Conv2d expects NHWC weight layout (out, kh, kw, in)
                if v.ndim == 4:
                    v = v.permute(0, 2, 3, 1)
                assert v.shape == our_sd[k].shape, \
                    f"Shape mismatch {k}: {v.shape} vs {our_sd[k].shape}"
                loaded[k] = mx.array(v.numpy())
        missing = set(our_sd) - set(loaded)
        unexpected = set(loaded) - set(our_sd)
        if missing:
            raise KeyError(f"[VAE-MLX] Missing keys: {missing}")
        if unexpected:
            raise KeyError(f"[VAE-MLX] Unexpected keys: {unexpected}")
        self.load_weights(list(loaded.items()), strict=False)


def load_flux2vae_mlx(path: str) -> Flux2VAEEncoder:
    model = Flux2VAEEncoder()
    model.load_safetensors(path)
    return model
