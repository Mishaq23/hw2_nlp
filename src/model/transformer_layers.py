"""Pre-LayerNorm encoder/decoder blocks with rotary self-attention."""

import torch
from torch import nn
from torch.nn import functional as F


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim, theta):
        super().__init__()
        inv_freq = theta ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, offset=0):
        positions = torch.arange(
            offset, offset + x.size(-2), device=x.device, dtype=torch.float32
        )
        angles = positions[:, None] * self.inv_freq.float()[None, :]
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos), -1
        ).flatten(-2)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, attention_dropout, bias, rope=None):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.attention_dropout = attention_dropout
        self.rope = rope
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def _split_heads(self, x):
        return x.reshape(x.size(0), x.size(1), self.n_heads, self.head_dim).transpose(
            1, 2
        )

    def project_kv(self, x):
        return self._split_heads(self.k_proj(x)), self._split_heads(self.v_proj(x))

    def forward(self, x, key_value=None, mask=None, cache=None):
        """Self-attention appends to cache; cross-attention reuses encoder K/V."""
        q = self._split_heads(self.q_proj(x))
        if key_value is None:
            k, v = self.project_kv(x)
            offset = 0 if cache is None else cache[0].size(-2)
            q, k = self.rope(q, offset), self.rope(k, offset)
            if cache is not None:
                k = torch.cat((cache[0], k), dim=-2)
                v = torch.cat((cache[1], v), dim=-2)
        else:
            k, v = self.project_kv(key_value) if cache is None else cache

        # SDPA uses True for allowed attention, matching our collate masks.
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(x.size(0), x.size(1), -1)
        return self.out_proj(attended), (k, v)


class FeedForward(nn.Sequential):
    def __init__(self, d_model, d_ff, dropout, activation, bias):
        activation_fn = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[activation]
        super().__init__(
            nn.Linear(d_model, d_ff, bias=bias),
            activation_fn(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model, bias=bias),
        )


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout,
        attention_dropout,
        activation,
        layer_norm_eps,
        bias,
        rope,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            d_model, n_heads, attention_dropout, bias, rope
        )
        self.ffn = FeedForward(d_model, d_ff, dropout, activation, bias)
        self.attention_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.ffn_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attended, _ = self.self_attention(self.attention_norm(x), mask=mask)
        x = x + self.dropout(attended)
        return x + self.dropout(self.ffn(self.ffn_norm(x)))


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        n_heads,
        d_ff,
        dropout,
        attention_dropout,
        activation,
        layer_norm_eps,
        bias,
        rope,
    ):
        super().__init__()
        self.self_attention = MultiHeadAttention(
            d_model, n_heads, attention_dropout, bias, rope
        )
        self.cross_attention = MultiHeadAttention(
            d_model, n_heads, attention_dropout, bias
        )
        self.ffn = FeedForward(d_model, d_ff, dropout, activation, bias)
        self.self_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.cross_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.ffn_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memory, self_mask, memory_mask, cache=None):
        self_cache, cross_cache = (None, None) if cache is None else cache
        attended, self_cache = self.self_attention(
            self.self_norm(x), mask=self_mask, cache=self_cache
        )
        x = x + self.dropout(attended)
        attended, cross_cache = self.cross_attention(
            self.cross_norm(x), key_value=memory, mask=memory_mask, cache=cross_cache
        )
        x = x + self.dropout(attended)
        x = x + self.dropout(self.ffn(self.ffn_norm(x)))
        return x, (self_cache, cross_cache)
