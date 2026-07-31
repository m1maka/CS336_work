from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .linear import Linear
from .rope import RotaryPositionalEmbedding


def softmax(x: Tensor, dim: int) -> Tensor:
    shifted = x - x.max(dim=dim, keepdim=True).values
    exponentials = torch.exp(shifted)
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key dimensions must match")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value sequence lengths must match")

    scores = query @ key.transpose(-1, -2)
    scores = scores / math.sqrt(query.shape[-1])
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("attention mask must be boolean")
        scores = scores.masked_fill(~mask, float("-inf"))

    return softmax(scores, dim=-1) @ value


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if (max_seq_len is None) != (theta is None):
            raise ValueError("max_seq_len and theta must either both be set or both be None")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = (
            RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device=device)
            if theta is not None and max_seq_len is not None
            else None
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        sequence_length = x.shape[-2]
        x = x.reshape(
            *x.shape[:-2], sequence_length, self.num_heads, self.head_dim
        )
        return x.transpose(-3, -2)

    def forward(
        self, x: Tensor, token_positions: Tensor | None = None
    ) -> Tensor:
        sequence_length = x.shape[-2]
        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)

        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=x.device,
        ).tril()
        attended = scaled_dot_product_attention(query, key, value, causal_mask)
        attended = attended.transpose(-3, -2).contiguous()
        attended = attended.reshape(*x.shape[:-2], sequence_length, self.d_model)
        return self.output_proj(attended)
