from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean cross-entropy computed without materializing probabilities."""
    if logits.ndim < 2 or logits.shape[:-1] != targets.shape:
        raise ValueError("targets must match every logits dimension except the final class dimension")

    shifted = logits - logits.max(dim=-1, keepdim=True).values
    log_normalizer = torch.log(torch.exp(shifted).sum(dim=-1))
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - target_logits).mean()


@torch.no_grad()
def clip_gradients(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    """Clip the global L2 norm of all present parameter gradients in place."""
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return
    total_norm = torch.linalg.vector_norm(
        torch.stack([torch.linalg.vector_norm(gradient.detach(), ord=2) for gradient in gradients]),
        ord=2,
    )
    coefficient = torch.clamp(max_l2_norm / (total_norm + 1e-6), max=1.0)
    for gradient in gradients:
        gradient.mul_(coefficient.to(device=gradient.device, dtype=gradient.dtype))
