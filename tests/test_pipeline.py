from __future__ import annotations

import numpy as np
import torch

from cs336_basics.generation import generate, sample_next_token
from cs336_basics.plotting import plot_metrics_svg
from cs336_basics.preprocess import load_token_array, tokenize_file
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.training import ModelConfig, TrainConfig, train
from cs336_basics.transformer_lm import TransformerLM


def _byte_tokenizer() -> BPETokenizer:
    return BPETokenizer({byte: bytes([byte]) for byte in range(256)}, [], ["<|endoftext|>"])


def test_tokenize_file_is_memory_mappable(tmp_path):
    text = "hello 🙃<|endoftext|>"
    input_path = tmp_path / "input.txt"
    output_path = tmp_path / "tokens.bin"
    input_path.write_text(text, encoding="utf-8")

    tokenizer = _byte_tokenizer()
    count = tokenize_file(tokenizer, input_path, output_path)
    tokens = load_token_array(output_path)

    assert count == len(tokens)
    assert tokenizer.decode(tokens.tolist()) == text


def test_top_p_can_restrict_sampling_to_argmax():
    torch.manual_seed(0)
    logits = torch.tensor([0.0, 1.0, 10.0])
    samples = [sample_next_token(logits, top_p=0.1).item() for _ in range(20)]
    assert samples == [2] * 20


def test_tiny_training_checkpoint_and_generation(tmp_path):
    model_config = ModelConfig(
        vocab_size=32,
        context_length=8,
        d_model=16,
        num_layers=1,
        num_heads=2,
        d_ff=32,
    )
    train_config = TrainConfig(
        batch_size=2,
        max_steps=2,
        warmup_steps=1,
        log_interval=1,
        eval_interval=2,
        eval_batches=1,
        checkpoint_interval=2,
    )
    data = np.arange(256, dtype=np.uint16) % model_config.vocab_size
    model, history = train(data, data, model_config, train_config, tmp_path)

    assert (tmp_path / "checkpoint.pt").exists()
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "metrics.csv").exists()
    assert any(row["split"] == "validation" for row in history)
    plot_metrics_svg([tmp_path / "metrics.csv"], tmp_path / "curve.svg")
    assert (tmp_path / "curve.svg").read_text(encoding="utf-8").startswith("<svg")

    output = generate(
        model,
        torch.tensor([1, 2], dtype=torch.long),
        max_new_tokens=3,
        context_length=model_config.context_length,
    )
    assert output.shape == (5,)


def test_architecture_ablation_variants_support_backward():
    tokens = torch.randint(0, 32, (2, 8))
    for norm_style, use_rope, ffn_type in (
        ("post", True, "swiglu"),
        ("none", True, "swiglu"),
        ("pre", False, "swiglu"),
        ("pre", True, "silu"),
    ):
        config = ModelConfig(
            vocab_size=32,
            context_length=8,
            d_model=16,
            num_layers=1,
            num_heads=2,
            d_ff=32,
            norm_style=norm_style,
            use_rope=use_rope,
            ffn_type=ffn_type,
        )
        model = TransformerLM(**vars(config))
        logits = model(tokens)
        assert logits.shape == (2, 8, 32)
        logits.sum().backward()
