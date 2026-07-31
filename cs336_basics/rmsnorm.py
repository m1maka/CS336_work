from __future__ import annotations

import torch
from torch import Tensor, nn


class RMSNorm(nn.Module):
    """Root mean square normalization over the final dimension."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if eps < 0:
            raise ValueError("eps must be non-negative")

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        rms = torch.sqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = x_float / rms
        return (normalized * self.weight.to(torch.float32)).to(input_dtype)
