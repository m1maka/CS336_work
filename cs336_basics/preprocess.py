from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from .tokenizer import BPETokenizer


def tokenize_file(
    tokenizer: BPETokenizer,
    input_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    dtype: str = "uint16",
    buffer_size: int = 1_000_000,
) -> int:
    """Stream a text corpus to a raw, memory-mappable token array."""
    token_dtype = np.dtype(dtype)
    if token_dtype.kind != "u":
        raise ValueError("token dtype must be an unsigned integer type")
    if max(tokenizer.id_to_token, default=0) > np.iinfo(token_dtype).max:
        raise ValueError(f"vocabulary IDs do not fit in {token_dtype}")
    if buffer_size <= 0:
        raise ValueError("buffer_size must be positive")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    buffer: list[int] = []
    with open(input_path, encoding="utf-8") as input_file, open(output_path, "wb") as output_file:
        for token_id in tokenizer.encode_iterable(input_file):
            buffer.append(token_id)
            if len(buffer) >= buffer_size:
                np.asarray(buffer, dtype=token_dtype).tofile(output_file)
                count += len(buffer)
                buffer.clear()
        if buffer:
            np.asarray(buffer, dtype=token_dtype).tofile(output_file)
            count += len(buffer)

    metadata = {"num_tokens": count, "dtype": token_dtype.name, "source": str(input_path)}
    with open(f"{output_path}.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
    return count


def load_token_array(path: str | os.PathLike[str], dtype: str = "uint16") -> np.ndarray:
    """Memory-map a .npy file or a raw token file created by tokenize_file."""
    path = Path(path)
    if path.suffix == ".npy":
        array = np.load(path, mmap_mode="r")
    else:
        array = np.memmap(path, dtype=np.dtype(dtype), mode="r")
    if array.ndim != 1:
        raise ValueError("token data must be one-dimensional")
    return array
