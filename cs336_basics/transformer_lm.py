from __future__ import annotations

import torch
from torch import Tensor, nn

from .embedding import Embedding
from .linear import Linear
from .rmsnorm import RMSNorm
from .transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        norm_style: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
    ) -> None:
        super().__init__()
        if context_length <= 0 or num_layers <= 0:
            raise ValueError("context_length and num_layers must be positive")

        self.context_length = context_length
        self.token_embeddings = Embedding(
            vocab_size, d_model, device=device, dtype=dtype
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                    norm_style=norm_style,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if norm_style != "none" else None
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        sequence_length = token_ids.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(
                f"Input length {sequence_length} exceeds context length "
                f"{self.context_length}"
            )

        positions = torch.arange(sequence_length, device=token_ids.device)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, positions)
        if self.ln_final is not None:
            x = self.ln_final(x)
        return self.lm_head(x)
