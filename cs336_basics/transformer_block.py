from __future__ import annotations

import torch
from torch import Tensor, nn

from .attention import MultiHeadSelfAttention
from .rmsnorm import RMSNorm
from .swiglu import SiLUFeedForward, SwiGLU


class TransformerBlock(nn.Module):
    """A pre-norm decoder-only Transformer block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm_style: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
    ) -> None:
        super().__init__()
        if norm_style not in {"pre", "post", "none"}:
            raise ValueError("norm_style must be 'pre', 'post', or 'none'")
        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError("ffn_type must be 'swiglu' or 'silu'")
        self.norm_style = norm_style
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len if use_rope else None,
            theta=theta if use_rope else None,
            device=device,
            dtype=dtype,
        )
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype) if norm_style != "none" else None
        self.ffn = (
            SwiGLU(d_model, d_ff, device=device, dtype=dtype)
            if ffn_type == "swiglu"
            else SiLUFeedForward(d_model, d_ff, device=device, dtype=dtype)
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype) if norm_style != "none" else None

    def forward(
        self, x: Tensor, token_positions: Tensor | None = None
    ) -> Tensor:
        if self.norm_style == "pre":
            assert self.ln1 is not None and self.ln2 is not None
            x = x + self.attn(self.ln1(x), token_positions)
            return x + self.ffn(self.ln2(x))

        x = x + self.attn(x, token_positions)
        if self.ln1 is not None:
            x = self.ln1(x)
        x = x + self.ffn(x)
        if self.ln2 is not None:
            x = self.ln2(x)
        return x
