"""DFlash draft model architecture for Qwen3.6-35B-A3B.

Matches the r0b0tlab/Qwen3.6-35B-A3B-DFlash architecture exactly.
5-layer non-causal transformer (paper: 5 layers best avg speedup).
KV injection: target features projected through draft's wk/wv, concatenated.
SwiGLU FFN. Shared/frozen target embedding + lm_head.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class Qwen3RMSNorm(nn.Module):
    """RMSNorm matching Qwen3's implementation (eps=1e-6)."""
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x / rms * self.weight.float()).to(dtype)


def rotate_half(x):
    """Rotate half the hidden dims of the input (for RoPE)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


class DFlashAttention(nn.Module):
    """Non-causal attention with KV injection from target features."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, rope_theta: float = 10000000.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_dim = num_heads * head_dim
        self.kv_dim = num_kv_heads * head_dim
        self.rope_theta = rope_theta

        self.q_proj = nn.Linear(hidden_size, self.q_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, self.kv_dim, bias=False)
        self.o_proj = nn.Linear(self.q_dim, hidden_size, bias=False)

        self.q_norm = Qwen3RMSNorm(head_dim)
        self.k_norm = Qwen3RMSNorm(head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,       # [q_len, hidden_size]
        target_hidden: torch.Tensor,       # [ctx_len, hidden_size]
        positions_q: torch.Tensor,         # [q_len] int64
        positions_k: torch.Tensor,         # [ctx_len + q_len] int64
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q_len, ctx_len = hidden_states.shape[0], target_hidden.shape[0]
        total_k = ctx_len + q_len
        device, dtype = hidden_states.device, hidden_states.dtype

        # Q from noise only
        q = self.q_proj(hidden_states)
        q = q.view(q_len, self.num_heads, self.head_dim)
        q = self.q_norm(q)

        # K/V from target features AND noise
        k_ctx = self.k_proj(target_hidden)      # [ctx_len, kv_dim]
        k_noise = self.k_proj(hidden_states)     # [q_len, kv_dim]
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)

        k = torch.cat([k_ctx, k_noise], dim=0)  # [total_k, kv_dim]
        v = torch.cat([v_ctx, v_noise], dim=0)

        k = k.view(total_k, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k)
        v = v.view(total_k, self.num_kv_heads, self.head_dim)

        # RoPE using HF-style apply_rotary_pos_emb
        cos_q, sin_q = self._compute_rope(positions_q, device, dtype)
        cos_k, sin_k = self._compute_rope(positions_k, device, dtype)
        q = self._apply_rope(q, cos_q, sin_q)
        k = self._apply_rope(k, cos_k, sin_k)

        # Transpose to [heads, seq, head_dim] for SDPA
        q = q.transpose(0, 1).unsqueeze(0)  # [1, n_head, q_len, head_dim]
        k = k.transpose(0, 1).unsqueeze(0)  # [1, n_kv_head, total_k, head_dim]
        v = v.transpose(0, 1).unsqueeze(0)

        # GQA: repeat KV heads
        if self.num_kv_heads < self.num_heads:
            n_groups = self.num_heads // self.num_kv_heads
            k = k.repeat_interleave(n_groups, dim=1)
            v = v.repeat_interleave(n_groups, dim=1)

        # Flash attention (non-causal)
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=False,
            scale=1.0 / math.sqrt(self.head_dim))

        attn_out = attn_out.squeeze(0).transpose(0, 1).reshape(q_len, -1)
        return self.o_proj(attn_out)

    def _compute_rope(self, positions, device, dtype):
        """Compute RoPE cos/sin for given positions using NEOX style."""
        # NEOX: freqs for dim//2, then interleave for pairs
        dim = self.head_dim
        half_dim = dim // 2
        theta = self.rope_theta
        freqs = 1.0 / (theta ** (torch.arange(0, half_dim, device=device).float() / half_dim))
        # positions: [seq_len] → [seq_len, 1]
        pos = positions.float().unsqueeze(1)
        # [seq_len, half_dim]
        emb = pos * freqs.unsqueeze(0)
        # Interleave for pairs: [seq_len, dim]
        emb = emb.repeat_interleave(2, dim=-1)
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def _apply_rope(self, x, cos, sin):
        """Apply RoPE: x * cos + rotate_half(x) * sin"""
        # x: [seq_len, num_heads, head_dim]
        # cos/sin: [seq_len, head_dim] → [seq_len, 1, head_dim]
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        return (x * cos) + (rotate_half(x) * sin)


class DFlashSwiGLUFFN(nn.Module):
    """SwiGLU feed-forward network."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 intermediate_size, rope_theta):
        super().__init__()
        self.input_layernorm = Qwen3RMSNorm(hidden_size)
        self.self_attn = DFlashAttention(
            hidden_size, num_heads, num_kv_heads, head_dim, rope_theta)
        self.post_attention_layernorm = Qwen3RMSNorm(hidden_size)
        self.mlp = DFlashSwiGLUFFN(hidden_size, intermediate_size)

    def forward(self, hidden_states, target_hidden, positions_q,
                positions_k, attention_mask=None):
        residual = hidden_states
        normed = self.input_layernorm(hidden_states)
        attn_out = self.self_attn(normed, target_hidden, positions_q,
                                  positions_k, attention_mask)
        hidden_states = residual + attn_out
        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states))
        return hidden_states


class DFlashDraftModel(nn.Module):
    """DFlash draft model: 5-layer non-causal transformer with KV injection.

    Input:
      - noise_embed: [q_len, hidden_size]
      - target_hidden_cat: [ctx_len, N*hidden_size]
    Output:
      - hidden_states: [q_len, hidden_size] (ready for lm_head projection)
    """

    def __init__(self, config: dict):
        super().__init__()
        hidden_size = config['hidden_size']
        self.hidden_size = hidden_size
        self.num_layers = config['num_draft_layers']
        self.n_target_features = config['n_target_features']
        num_heads = config['num_attention_heads']
        num_kv_heads = config['num_key_value_heads']
        head_dim = config['head_dim']
        intermediate_size = config['intermediate_size']
        rope_theta = config['rope_theta']

        self.fc = nn.Linear(
            self.n_target_features * hidden_size, hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(hidden_size)

        self.layers = nn.ModuleList([
            DFlashDecoderLayer(
                hidden_size, num_heads, num_kv_heads, head_dim,
                intermediate_size, rope_theta)
            for _ in range(self.num_layers)
        ])
        self.out_norm = Qwen3RMSNorm(hidden_size)

    def fuse_target_features(self, target_hidden_cat):
        return self.hidden_norm(self.fc(target_hidden_cat))

    def forward(self, noise_embed, target_hidden_cat, positions_q,
                positions_k, attention_mask=None):
        # target_hidden_cat: [ctx_len+q_len, N*hidden] — includes query positions too
        # But attention only needs the context portion [0:ctx_len]
        ctx_len = positions_k.shape[0] - positions_q.shape[0]
        target_feat = self.fuse_target_features(target_hidden_cat[:ctx_len])
        hidden = noise_embed
        for layer in self.layers:
            hidden = layer(hidden, target_feat, positions_q,
                           positions_k, attention_mask)
        return self.out_norm(hidden)
