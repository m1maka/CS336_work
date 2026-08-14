from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


class AdamW(torch.optim.Optimizer):
    """Adam with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0 or eps < 0 or weight_decay < 0:
            raise ValueError("lr, eps, and weight_decay must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("betas must lie in [0, 1)")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            learning_rate = group["lr"]
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                parameter.mul_(1 - learning_rate * group["weight_decay"])
                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = learning_rate * math.sqrt(bias_correction2) / bias_correction1
                parameter.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(group["eps"]), value=-step_size)

        return loss


def cosine_learning_rate(
    iteration: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Linear warmup followed by cosine decay to a fixed minimum."""
    if iteration < 0 or warmup_iters < 0 or cosine_cycle_iters < warmup_iters:
        raise ValueError("invalid iteration or schedule boundaries")
    if min_learning_rate < 0 or max_learning_rate < min_learning_rate:
        raise ValueError("learning rates must satisfy max >= min >= 0")
    if iteration < warmup_iters:
        return max_learning_rate * iteration / warmup_iters
    if iteration >= cosine_cycle_iters:
        return min_learning_rate
    progress = (iteration - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)
