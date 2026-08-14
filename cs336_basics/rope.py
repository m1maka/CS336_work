from __future__ import annotations

import torch
from torch import Tensor, nn


class RotaryPositionalEmbedding(nn.Module):
    """Apply pairwise rotary position embeddings to queries or keys."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        if theta <= 0:
            raise ValueError("theta must be positive")
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("d_k must be a positive even integer")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        frequencies = theta ** (
            -torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        )
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * frequencies[None, :]
        self.cos: Tensor
        self.sin: Tensor
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

        self.d_k = d_k
        self.max_seq_len = max_seq_len

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError(
                f"Expected final dimension {self.d_k}, got {x.shape[-1]}"
            )
        if token_positions.shape[-1] != x.shape[-2]:
            raise ValueError("token_positions must match the input sequence length")
        if token_positions.numel() and (
            token_positions.min() < 0 or token_positions.max() >= self.max_seq_len
        ):
            raise ValueError("token position is outside the precomputed RoPE range")

        token_positions = token_positions.to(device=self.cos.device, dtype=torch.long)
        cos = self.cos[token_positions].to(x.dtype)
        sin = self.sin[token_positions].to(x.dtype)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        pairs = x.reshape(*x.shape[:-1], self.d_k // 2, 2)
        first = pairs[..., 0] * cos - pairs[..., 1] * sin
        second = pairs[..., 0] * sin + pairs[..., 1] * cos
        return torch.stack((first, second), dim=-1).flatten(-2)
