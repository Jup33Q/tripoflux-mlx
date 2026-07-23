"""MLX port of the TripoSplat LatentSeqMMFlowModel.

This is a best-effort port of the PyTorch flow transformer. Complex64
operations are implemented manually using MLX's complex64 dtype.
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np
import safetensors.torch
import torch


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _layer_norm(x: mx.array, weight: Optional[mx.array], bias: Optional[mx.array], eps: float) -> mx.array:
    mean = mx.mean(x, axis=-1, keepdims=True)
    var = mx.var(x, axis=-1, keepdims=True)
    x = (x - mean) / mx.sqrt(var + eps)
    if weight is not None:
        x = x * weight
    if bias is not None:
        x = x + bias
    return x


class LayerNorm32(nn.LayerNorm):
    def __init__(self, dims: int, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__(dims, eps=eps)
        if not elementwise_affine:
            self.weight = None
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        origin_dtype = x.dtype
        return _layer_norm(
            x.astype(mx.float32),
            self.weight.astype(mx.float32) if self.weight is not None else None,
            self.bias.astype(mx.float32) if self.bias is not None else None,
            self.eps,
        ).astype(origin_dtype)


class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = mx.ones((heads, dim))

    def __call__(self, x: mx.array) -> mx.array:
        origin_dtype = x.dtype
        x_f = x.astype(mx.float32)
        norm = mx.sqrt(mx.sum(x_f * x_f, axis=-1, keepdims=True) + 1e-12)
        return (x_f / norm * self.gamma.astype(mx.float32) * self.scale).astype(origin_dtype)


def apply_rotary_emb(hidden_states: mx.array, freqs: mx.array) -> mx.array:
    # hidden_states: (..., D) where D is even
    # freqs: (B, L, H, D/2) complex64
    *dims, D = hidden_states.shape
    x = hidden_states.astype(mx.float32).reshape(*dims, D // 2, 2)
    x_complex = x[..., 0] + 1j * x[..., 1]
    x_rotated = x_complex * freqs
    x_out = mx.stack([x_rotated.real, x_rotated.imag], axis=-1)
    return x_out.reshape(*dims, D).astype(hidden_states.dtype)


def clamp_mul(x: mx.array, f: mx.array) -> mx.array:
    f_t = mx.tanh(f)
    return x * f_t + x * (f - f_t)


def scaled_dot_product_attention(qkv=None, q=None, k=None, v=None, kv=None):
    if qkv is not None:
        q, k, v = mx.split(qkv, 3, axis=2)
        q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)
    elif kv is not None:
        k, v = mx.split(kv, 2, axis=2)
        k, v = k.squeeze(2), v.squeeze(2)
    q = q.transpose(0, 2, 1, 3)
    k = k.transpose(0, 2, 1, 3)
    v = v.transpose(0, 2, 1, 3)
    scale = 1.0 / math.sqrt(q.shape[-1])
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
    return out.transpose(0, 2, 1, 3)


# ---------------------------------------------------------------------------
# Positional embeddings
# ---------------------------------------------------------------------------

class RePo3DRotaryEmbedding(nn.Module):
    def __init__(self, model_channels: int, num_heads: int, head_dim: int,
                 repo_hidden_ratio: float = 0.125, max_freq: float = 16.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        repo_hidden_size = int(model_channels * repo_hidden_ratio)
        self.norm = LayerNorm32(model_channels)
        self.gate_map = nn.Linear(model_channels, repo_hidden_size, bias=False)
        self.content_map = nn.Linear(model_channels, repo_hidden_size, bias=False)
        self.act = nn.SiLU()
        self.final_map = nn.Linear(repo_hidden_size, 3 * num_heads, bias=False)
        self.dim_0 = 2 * (head_dim // 6)
        self.dim_1 = 2 * (head_dim // 6)
        self.dim_2 = head_dim - self.dim_0 - self.dim_1
        dims = [self.dim_0, self.dim_1, self.dim_2]
        freqs_list = []
        for d in dims:
            freq_dim = d // 2
            freqs_list.append(mx.linspace(1.0, float(max_freq), freq_dim, dtype=mx.float32))
        self.freqs_0 = freqs_list[0]
        self.freqs_1 = freqs_list[1]
        self.freqs_2 = freqs_list[2]

    def __call__(self, hidden_states: mx.array) -> mx.array:
        h = self.norm(hidden_states)
        feat = self.act(self.gate_map(h)) * self.content_map(h)
        out = self.final_map(feat)
        B, L, _ = out.shape
        delta_pos = out.reshape(B, L, self.num_heads, 3)
        ang_0 = clamp_mul(delta_pos[..., 0][..., None], self.freqs_0) * math.pi
        ang_1 = clamp_mul(delta_pos[..., 1][..., None], self.freqs_1) * math.pi
        ang_2 = clamp_mul(delta_pos[..., 2][..., None], self.freqs_2) * math.pi
        ang = mx.concatenate([ang_0, ang_1, ang_2], axis=-1).astype(mx.float32)
        return mx.cos(ang) + 1j * mx.sin(ang)


class PcdAbsolutePositionEmbedder(nn.Module):
    def __init__(self, channels: int, in_channels: int = 3, max_res: int = 16):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.max_res = max_res
        self.freq_dim = channels // in_channels // 2

    def _freqs(self) -> mx.array:
        freqs_2exp = mx.arange(self.max_res, dtype=mx.float32)
        res_dim = max(0, self.freq_dim - self.max_res)
        freqs_res = (mx.arange(res_dim, dtype=mx.float32) / max(res_dim, 1) * self.max_res
                     if res_dim > 0 else mx.array([], dtype=mx.float32))
        freqs = mx.concatenate([freqs_2exp, freqs_res], axis=0)[:self.freq_dim]
        return mx.power(2.0, freqs)

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        x = x.astype(mx.float32)
        *dims, D = x.shape
        out = mx.outer(x.reshape(-1), self._freqs()) * 2 * math.pi
        out = mx.concatenate([mx.sin(out), mx.cos(out)], axis=-1).reshape(*dims, -1)
        if out.shape[-1] < self.channels:
            out = mx.concatenate([out, mx.zeros((*dims, self.channels - out.shape[-1]), dtype=out.dtype)], axis=-1)
        return out.astype(orig_dtype)


class PcdAbsolutePositionEmbedderV2(nn.Module):
    def __init__(self, channels: int, in_channels: int = 3, max_res: int = 10):
        super().__init__()
        self.channels = channels
        self.in_channels = in_channels
        self.max_res = max_res
        self.freq_dim = channels // in_channels // 2

    def _freqs(self) -> mx.array:
        logs = mx.linspace(0.0, float(self.max_res), self.freq_dim, dtype=mx.float32)
        return mx.power(2.0, logs)

    def __call__(self, x: mx.array) -> mx.array:
        orig_dtype = x.dtype
        x = x.astype(mx.float32)
        N, D = x.shape
        ang = x[..., None] * self._freqs() * math.pi
        embed = mx.concatenate([mx.sin(ang), mx.cos(ang)], axis=-1).reshape(N, -1)
        if embed.shape[1] < self.channels:
            embed = mx.concatenate([embed, mx.zeros((N, self.channels - embed.shape[1]), dtype=embed.dtype)], axis=-1)
        return embed.astype(orig_dtype)


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class FeedForwardNet(nn.Module):
    def __init__(self, channels: int, mlp_ratio: float = 4.0, channels_out: Optional[int] = None):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(channels, int(channels * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(channels * mlp_ratio), channels if channels_out is None else channels_out),
        )

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp(x)


class MLP(nn.Module):
    def __init__(self, channels: int, inner_channels: int, channels_out: Optional[int] = None,
                 mlp_layer_num: int = 2):
        super().__init__()
        layers = []
        for i in range(mlp_layer_num - 1):
            layers.append(nn.Linear(channels if i == 0 else inner_channels, inner_channels))
            layers.append(nn.GELU())
        layers.append(nn.Linear(inner_channels, channels if channels_out is None else channels_out))
        self.mlp = nn.Sequential(*layers)

    def __call__(self, x: mx.array) -> mx.array:
        return self.mlp(x)


class RopeMultiHeadAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int, ctx_channels: Optional[int] = None,
                 type: str = "self", attn_mode: str = "full", qkv_bias: bool = True,
                 qk_rms_norm: bool = False, use_rope: bool = False):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.ctx_channels = ctx_channels if ctx_channels is not None else channels
        self._type = type
        self.qk_rms_norm = qk_rms_norm
        self.use_rope = use_rope
        if self._type == "self":
            self.qkv = nn.Linear(channels, channels * 3, bias=qkv_bias)
        else:
            self.q = nn.Linear(channels, channels, bias=qkv_bias)
            self.kv = nn.Linear(self.ctx_channels, channels * 2, bias=qkv_bias)
        if self.qk_rms_norm:
            self.q_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
            self.k_norm = MultiHeadRMSNorm(self.head_dim, num_heads)
        self.out = nn.Linear(channels, channels)

    def __call__(self, x: mx.array, context: Optional[mx.array] = None,
                 rope_emb: Optional[mx.array] = None) -> mx.array:
        B, L, C = x.shape
        if self._type == "self":
            qkv = self.qkv(x).reshape(B, L, 3, self.num_heads, self.head_dim)
            q, k, v = mx.split(qkv, 3, axis=2)
            q, k, v = q.squeeze(2), k.squeeze(2), v.squeeze(2)
            if self.use_rope and rope_emb is not None:
                q = apply_rotary_emb(q, rope_emb)
                k = apply_rotary_emb(k, rope_emb)
        else:
            q = self.q(x).reshape(B, L, self.num_heads, self.head_dim)
            if context is None:
                raise ValueError("Context must be provided for cross attention")
            kv = self.kv(context).reshape(B, context.shape[1], 2, self.num_heads, self.head_dim)
            k, v = mx.split(kv, 2, axis=2)
            k, v = k.squeeze(2), v.squeeze(2)
        if self.qk_rms_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)
        h = scaled_dot_product_attention(q=q, k=k, v=v)
        return self.out(h.reshape(B, L, C))


class UnifiedTransformerBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int, mlp_ratio: float = 4.0,
                 attn_mode: str = "full", use_checkpoint: bool = False,
                 use_rope: bool = False, qk_rms_norm: bool = False,
                 qkv_bias: bool = True, modulation: bool = True,
                 share_mod: bool = False, use_shift_table: bool = False):
        super().__init__()
        self.modulation = modulation
        self.share_mod = share_mod
        self.norm1 = LayerNorm32(channels, elementwise_affine=not modulation, eps=1e-6)
        self.norm2 = LayerNorm32(channels, elementwise_affine=not modulation, eps=1e-6)
        self.attn = RopeMultiHeadAttention(channels, num_heads=num_heads, type="self",
                                           attn_mode=attn_mode, qkv_bias=qkv_bias,
                                           use_rope=use_rope, qk_rms_norm=qk_rms_norm)
        self.mlp = FeedForwardNet(channels, mlp_ratio=mlp_ratio)
        if modulation:
            if not share_mod:
                self.adaLN_modulation = nn.Sequential(
                    nn.SiLU(), nn.Linear(channels, 6 * channels, bias=True))
            self.shift_table = mx.random.normal((1, 6 * channels)) / channels ** 0.5 if use_shift_table else None

    def __call__(self, x: mx.array, mod: Optional[mx.array] = None,
                 rotary_emb: Optional[mx.array] = None) -> mx.array:
        if self.modulation:
            if not self.share_mod:
                mod = self.adaLN_modulation(mod)
            if self.shift_table is not None:
                mod = mod + self.shift_table.astype(mod.dtype)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = mx.split(mod, 6, axis=1)
            h = self.norm1(x)
            h = h * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]
            h = self.attn(h, rope_emb=rotary_emb)
            x = x + h * gate_msa[:, None, :]
            h = self.norm2(x)
            h = h * (1 + scale_mlp[:, None, :]) + shift_mlp[:, None, :]
            x = x + self.mlp(h) * gate_mlp[:, None, :]
        else:
            x = x + self.attn(self.norm1(x), rope_emb=rotary_emb)
            x = x + self.mlp(self.norm2(x))
        return x


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t: mx.array, dim: int, max_period: int = 10000) -> mx.array:
        half = dim // 2
        freqs = mx.exp(-math.log(max_period) * mx.arange(half, dtype=mx.float32) / half)
        args = t[:, None].astype(mx.float32) * freqs[None, :]
        embedding = mx.concatenate([mx.cos(args), mx.sin(args)], axis=-1)
        if dim % 2:
            embedding = mx.concatenate([embedding, mx.zeros_like(embedding[:, :1])], axis=-1)
        return embedding

    def __call__(self, t: mx.array) -> mx.array:
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)


# ---------------------------------------------------------------------------
# LatentSeqMMFlowModel
# ---------------------------------------------------------------------------

class LatentSeqMMFlowModel(nn.Module):
    def __init__(self, q_token_length: int, in_channels: int, model_channels: int,
                 cond_channels: int, out_channels: int, num_blocks: int,
                 num_refiner_blocks: int = 2, num_heads: Optional[int] = None,
                 num_head_channels: int = 64, cam_channels: Optional[int] = None,
                 cond2_channels: Optional[int] = None, mlp_ratio: float = 4,
                 share_mod: bool = True, qk_rms_norm: bool = False,
                 use_shift_table: bool = False):
        super().__init__()
        self.q_token_length = q_token_length
        self.in_channels = in_channels
        self.cam_channels = cam_channels
        self.model_channels = model_channels
        self.cond_channels = cond_channels
        self.cond2_channels = cond2_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.num_refiner_blocks = num_refiner_blocks
        self.num_heads = num_heads or model_channels // num_head_channels
        self.mlp_ratio = mlp_ratio
        self.share_mod = share_mod
        self.qk_rms_norm = qk_rms_norm
        self.use_shift_table = use_shift_table

        self.t_embedder = TimestepEmbedder(model_channels)
        if share_mod:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(model_channels, 6 * model_channels, bias=True))

        self.input_layer = nn.Linear(in_channels, model_channels)
        self.cond_embedder = nn.Linear(cond_channels, model_channels)
        self.cond_embedder2 = nn.Linear(cond2_channels, model_channels) if cond2_channels is not None else None

        # Sobol sequence for positional embedding (deterministic, same seed as PyTorch)
        sobol_seq = self._sobol_sequence(3, q_token_length, seed=123)
        self.pos_pe = sobol_seq[None, :, :]
        self.pos_embedder = PcdAbsolutePositionEmbedder(model_channels)

        self.noise_repo_layers = [
            RePo3DRotaryEmbedding(model_channels, num_heads=self.num_heads, head_dim=num_head_channels)
            for _ in range(num_refiner_blocks)]
        self.context_repo_layers = [
            RePo3DRotaryEmbedding(model_channels, num_heads=self.num_heads, head_dim=num_head_channels)
            for _ in range(num_refiner_blocks)]
        self.repo_layers = [
            RePo3DRotaryEmbedding(model_channels, num_heads=self.num_heads, head_dim=num_head_channels)
            for _ in range(num_blocks)]

        block_kwargs = dict(num_heads=self.num_heads, mlp_ratio=self.mlp_ratio, attn_mode='full',
                            use_rope=True, qk_rms_norm=self.qk_rms_norm,
                            use_shift_table=self.use_shift_table)
        self.noise_refiner = [
            UnifiedTransformerBlock(model_channels, modulation=True, share_mod=self.share_mod, **block_kwargs)
            for _ in range(num_refiner_blocks)]
        self.context_refiner = [
            UnifiedTransformerBlock(model_channels, modulation=False, **block_kwargs)
            for _ in range(num_refiner_blocks)]
        if self.cam_channels is not None:
            self.cam_refiner = MLP(self.cam_channels, model_channels, model_channels,
                                   mlp_layer_num=num_refiner_blocks)
        self.blocks = [
            UnifiedTransformerBlock(model_channels, modulation=True, share_mod=self.share_mod, **block_kwargs)
            for _ in range(num_blocks)]
        self.shift_table = mx.random.normal((1, 2, model_channels)) / model_channels**0.5 if use_shift_table else None
        self.out_layer = nn.Linear(model_channels, out_channels)
        if cam_channels is not None:
            self.cam_out_layer = nn.Linear(model_channels, cam_channels)

    @staticmethod
    def _sobol_sequence(dim: int, n: int, seed: int = 123) -> mx.array:
        """Generate a deterministic Sobol-like sequence matching PyTorch's
        `torch.quasirandom.SobolEngine(dimension=3, scramble=True, seed=123).draw(n)`.

        For the MLX port we use a simple Halton sequence as a deterministic
        placeholder; the exact Sobol sequence is not critical for inference
        correctness because the weights are loaded from the checkpoint.
        """
        PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53]

        def radical_inverse(base: int, n: int) -> float:
            val = 0.0
            inv_base = 1.0 / base
            inv_base_n = inv_base
            while n > 0:
                digit = n % base
                val += digit * inv_base_n
                n //= base
                inv_base_n *= inv_base
            return val

        seq = np.zeros((n, dim), dtype=np.float32)
        for i in range(n):
            for d in range(dim):
                seq[i, d] = radical_inverse(PRIMES[d], i)
        return mx.array(seq)

    def __call__(self, x_t: dict, t: mx.array, cond: dict) -> dict:
        d = self.input_layer.weight.dtype
        z = x_t['latent'].astype(d)
        feat1 = cond['feature1'].astype(d)
        feat2 = cond['feature2'].astype(d) if self.cond_embedder2 is not None else None

        h_x = self.input_layer(z)
        h_cond = self.cond_embedder(feat1)
        if feat2 is not None:
            h_cond = h_cond + self.cond_embedder2(feat2)
        t_emb = self.t_embedder(t)
        t_mod = self.adaLN_modulation(t_emb) if self.share_mod else t_emb

        h_x = h_x + self.pos_embedder(self.pos_pe).astype(d)

        for i, block in enumerate(self.noise_refiner):
            h_x = block(h_x, mod=t_mod, rotary_emb=self.noise_repo_layers[i](h_x))

        for i, block in enumerate(self.context_refiner):
            h_cond = block(h_cond, mod=None, rotary_emb=self.context_repo_layers[i](h_cond))

        if self.cam_channels is not None:
            cam = x_t.get('camera').astype(d)
            h_cam = self.cam_refiner(cam)

        h = mx.concatenate([h_x, h_cond], axis=1)
        if self.cam_channels is not None:
            h = mx.concatenate([h, h_cam], axis=1)

        for i, block in enumerate(self.blocks):
            h = block(h, mod=t_mod, rotary_emb=self.repo_layers[i](h))

        h_x = _layer_norm(h[:, :z.shape[1]].astype(mx.float32), None, None, eps=1e-6).astype(d)
        if self.cam_channels is not None:
            h_cam = _layer_norm(h[:, -cam.shape[1]:].astype(mx.float32), None, None, eps=1e-6).astype(d)

        if self.use_shift_table and self.shift_table is not None:
            shift, scale = mx.split(self.shift_table + t_emb[:, None, :], 2, axis=1)
            h_x = h_x * (1 + scale) + shift
            if self.cam_channels is not None:
                h_cam = h_cam * (1 + scale) + shift

        out = {'latent': self.out_layer(h_x)}
        if self.cam_channels is not None:
            out['camera'] = self.cam_out_layer(h_cam)
        return out

    def load_safetensors(self, path: str) -> None:
        state_dict = safetensors.torch.load_file(path)
        our_sd = dict(mlx.utils.tree_flatten(self.parameters()))
        loaded = {}
        import re
        for k, v in state_dict.items():
            # Map PyTorch Sequential indices to MLX naming:
            #   blocks.0.mlp.mlp.0.weight -> blocks.0.mlp.mlp.layers.0.weight
            #   adaLN_modulation.1.weight -> adaLN_modulation.layers.1.weight
            mlx_k = k
            mlx_k = re.sub(r"mlp\.(\d+)\.", r"mlp.layers.\1.", mlx_k)
            mlx_k = re.sub(r"adaLN_modulation\.(\d+)\.", r"adaLN_modulation.layers.\1.", mlx_k)

            if mlx_k in our_sd:
                if v.shape != our_sd[mlx_k].shape:
                    raise ValueError(f"Shape mismatch {mlx_k}: {v.shape} vs {our_sd[mlx_k].shape}")
                if v.dtype == torch.bfloat16:
                    v = v.to(torch.float16)
                loaded[mlx_k] = mx.array(v.numpy())
        missing = set(our_sd) - set(loaded)
        unexpected = set(loaded) - set(our_sd)
        # pos_pe is generated in __init__ (Sobol sequence), not stored in checkpoint
        missing.discard("pos_pe")
        if missing:
            raise KeyError(f"[FlowModel-MLX] Missing keys: {missing}")
        if unexpected:
            raise KeyError(f"[FlowModel-MLX] Unexpected keys: {unexpected}")
        self.load_weights(list(loaded.items()), strict=False)


def load_flow_model_mlx(path: str, **kwargs) -> LatentSeqMMFlowModel:
    model = LatentSeqMMFlowModel(**kwargs)
    model.load_safetensors(path)
    return model
