from __future__ import annotations

import csv
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn

from .checkpoint import load_checkpoint, save_checkpoint
from .data import get_batch
from .nn_utils import clip_gradients, cross_entropy
from .optimizer import AdamW, cosine_learning_rate
from .preprocess import load_token_array
from .transformer_lm import TransformerLM


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 10_000
    context_length: int = 256
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 16
    d_ff: int = 1344
    rope_theta: float = 10_000.0
    norm_style: str = "pre"
    use_rope: bool = True
    ffn_type: str = "swiglu"


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 32
    max_steps: int = 5_000
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    max_grad_norm: float = 1.0
    log_interval: int = 10
    eval_interval: int = 250
    eval_batches: int = 20
    checkpoint_interval: int = 500
    seed: int = 42


class MetricsLogger:
    def __init__(self, output_dir: Path, resume: bool = False) -> None:
        self.path = output_dir / "metrics.csv"
        write_header = not resume or not self.path.exists()
        self.file = open(self.path, "a" if resume else "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=["iteration", "elapsed_seconds", "split", "loss", "perplexity", "learning_rate"],
        )
        if write_header:
            self.writer.writeheader()
            self.file.flush()

    def log(self, iteration: int, elapsed: float, split: str, loss: float, learning_rate: float) -> None:
        perplexity = math.exp(loss) if loss < 80 else float("inf")
        self.writer.writerow(
            {
                "iteration": iteration,
                "elapsed_seconds": f"{elapsed:.3f}",
                "split": split,
                "loss": f"{loss:.6f}",
                "perplexity": f"{perplexity:.6f}",
                "learning_rate": f"{learning_rate:.8g}",
            }
        )
        self.file.flush()

    def close(self) -> None:
        self.file.close()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: np.ndarray,
    *,
    batch_size: int,
    context_length: int,
    num_batches: int,
    device: torch.device,
) -> float:
    if num_batches <= 0:
        raise ValueError("num_batches must be positive")
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(num_batches):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        losses.append(cross_entropy(model(inputs), targets).item())
    model.train(was_training)
    return float(np.mean(losses))


def train(
    train_data: np.ndarray,
    validation_data: np.ndarray,
    model_config: ModelConfig,
    train_config: TrainConfig,
    output_dir: str | os.PathLike[str],
    *,
    device: str = "cpu",
    resume_from: str | os.PathLike[str] | None = None,
    compile_model: bool = False,
) -> tuple[TransformerLM, list[dict[str, float | int | str]]]:
    """Train a Transformer LM and persist metrics/config/checkpoint artifacts."""
    if train_config.max_steps <= 0 or train_config.batch_size <= 0:
        raise ValueError("max_steps and batch_size must be positive")
    if not 0 <= train_config.warmup_steps <= train_config.max_steps:
        raise ValueError("warmup_steps must lie between zero and max_steps")
    if min(train_config.log_interval, train_config.eval_interval, train_config.checkpoint_interval) <= 0:
        raise ValueError("logging, evaluation, and checkpoint intervals must be positive")
    if len(train_data) <= model_config.context_length or len(validation_data) <= model_config.context_length:
        raise ValueError("training and validation data must be longer than the context length")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    torch_device = torch.device(device)
    random.seed(train_config.seed)
    np.random.seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    model = TransformerLM(**asdict(model_config), device=torch_device)
    optimizer = AdamW(
        model.parameters(),
        lr=train_config.max_learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        eps=train_config.eps,
        weight_decay=train_config.weight_decay,
    )
    start_iteration = 0
    if resume_from is not None:
        start_iteration = load_checkpoint(resume_from, model, optimizer)

    with open(output_path / "config.json", "w", encoding="utf-8") as config_file:
        json.dump(
            {"model": asdict(model_config), "training": asdict(train_config), "device": device},
            config_file,
            indent=2,
        )

    if compile_model:
        compiled = torch.compile(model, backend="aot_eager") if torch_device.type == "mps" else torch.compile(model)
        training_model = cast(nn.Module, compiled)
    else:
        training_model = model
    logger = MetricsLogger(output_path, resume=resume_from is not None)
    history: list[dict[str, float | int | str]] = []
    started_at = time.perf_counter()
    try:
        for iteration in range(start_iteration, train_config.max_steps):
            learning_rate = cosine_learning_rate(
                iteration,
                train_config.max_learning_rate,
                train_config.min_learning_rate,
                train_config.warmup_steps,
                train_config.max_steps,
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            inputs, targets = get_batch(
                train_data,
                train_config.batch_size,
                model_config.context_length,
                torch_device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = cross_entropy(training_model(inputs), targets)
            loss.backward()
            clip_gradients(model.parameters(), train_config.max_grad_norm)
            optimizer.step()

            completed_steps = iteration + 1
            elapsed = time.perf_counter() - started_at
            if completed_steps == 1 or completed_steps % train_config.log_interval == 0:
                value = loss.item()
                logger.log(completed_steps, elapsed, "train", value, learning_rate)
                history.append({"iteration": completed_steps, "split": "train", "loss": value})
                print(f"step={completed_steps} train_loss={value:.4f} lr={learning_rate:.3g} elapsed={elapsed:.1f}s")

            if completed_steps % train_config.eval_interval == 0 or completed_steps == train_config.max_steps:
                value = evaluate(
                    training_model,
                    validation_data,
                    batch_size=train_config.batch_size,
                    context_length=model_config.context_length,
                    num_batches=train_config.eval_batches,
                    device=torch_device,
                )
                elapsed = time.perf_counter() - started_at
                logger.log(completed_steps, elapsed, "validation", value, learning_rate)
                history.append({"iteration": completed_steps, "split": "validation", "loss": value})
                print(f"step={completed_steps} validation_loss={value:.4f} perplexity={math.exp(value):.2f}")

            if completed_steps % train_config.checkpoint_interval == 0 or completed_steps == train_config.max_steps:
                save_checkpoint(model, optimizer, completed_steps, output_path / "checkpoint.pt")
    finally:
        logger.close()

    return model, history


def train_from_files(
    train_path: str | os.PathLike[str],
    validation_path: str | os.PathLike[str],
    *,
    data_dtype: str,
    **kwargs,
):
    train_data = load_token_array(train_path, data_dtype)
    validation_data = load_token_array(validation_path, data_dtype)
    return train(train_data, validation_data, **kwargs)
