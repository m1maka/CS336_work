from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample next-token prediction examples from a one-dimensional token array."""
    if dataset.ndim != 1:
        raise ValueError("dataset must be one-dimensional")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive")
    if len(dataset) <= context_length:
        raise ValueError("dataset must contain more than context_length tokens")

    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    offsets = np.arange(context_length + 1)
    sequences = np.asarray(dataset[starts[:, None] + offsets[None, :]], dtype=np.int64)
    batch = torch.from_numpy(sequences).to(device=device)
    return batch[:, :-1], batch[:, 1:]
