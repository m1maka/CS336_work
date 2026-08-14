from __future__ import annotations

import torch
from torch import Tensor, nn

from .linear import Linear


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def default_d_ff(d_model: int) -> int:
    """Round the canonical 8/3 expansion up to a hardware-friendly multiple."""
    return 64 * ((8 * d_model + 3 * 64 - 1) // (3 * 64))


class SwiGLU(nn.Module):
    """Position-wise gated feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        d_ff = default_d_ff(d_model) if d_ff is None else d_ff
        if d_ff <= 0:
            raise ValueError("d_ff must be positive")

        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFeedForward(nn.Module):
    """Two-matrix SiLU feed-forward network used by the assignment ablation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_ff <= 0:
            raise ValueError("d_ff must be positive")
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))
