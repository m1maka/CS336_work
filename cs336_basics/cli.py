from __future__ import annotations

import argparse
import json
import pickle
import resource
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import get_type_hints

import torch

from .BPE_train import train_bpe
from .generation import generate
from .plotting import plot_metrics_svg
from .preprocess import tokenize_file
from .tokenizer import BPETokenizer
from .training import ModelConfig, TrainConfig, train_from_files
from .transformer_lm import TransformerLM


def _add_dataclass_arguments(parser: argparse.ArgumentParser, dataclass_type, prefix: str = "") -> None:
    type_hints = get_type_hints(dataclass_type)
    for field in fields(dataclass_type):
        option = f"--{prefix}{field.name.replace('_', '-')}"
        argument_type = _parse_bool if type_hints[field.name] is bool else type_hints[field.name]
        parser.add_argument(option, type=argument_type, default=field.default)


def _parse_bool(value: str) -> bool:
    normalized = value.lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError("expected true or false")


def _dataclass_from_args(dataclass_type, args, prefix: str = ""):
    return dataclass_type(**{field.name: getattr(args, f"{prefix}{field.name}") for field in fields(dataclass_type)})


def _load_tokenizer(args) -> BPETokenizer:
    return BPETokenizer.from_files(args.vocab, args.merges, args.special_token)


def _write_or_print_json(data: dict, output_path: str | None) -> None:
    rendered = json.dumps(data, indent=2, ensure_ascii=False)
    if output_path is None:
        print(rendered)
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CS336 Assignment 1 utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bpe = subparsers.add_parser("train-bpe", help="train and serialize a BPE tokenizer")
    bpe.add_argument("--input", required=True)
    bpe.add_argument("--vocab-size", type=int, required=True)
    bpe.add_argument("--special-token", action="append", default=[])
    bpe.add_argument("--vocab-out", required=True)
    bpe.add_argument("--merges-out", required=True)
    bpe.add_argument("--num-processes", type=int)
    bpe.add_argument("--stats-out")

    tokenize = subparsers.add_parser("tokenize", help="encode text to a raw memory-mappable token file")
    tokenize.add_argument("--input", required=True)
    tokenize.add_argument("--output", required=True)
    tokenize.add_argument("--vocab", required=True)
    tokenize.add_argument("--merges", required=True)
    tokenize.add_argument("--special-token", action="append", default=[])
    tokenize.add_argument("--dtype", default="uint16")

    tokenizer_stats = subparsers.add_parser("tokenizer-stats", help="measure a tokenizer's compression ratio")
    tokenizer_stats.add_argument("--input", required=True)
    tokenizer_stats.add_argument("--vocab", required=True)
    tokenizer_stats.add_argument("--merges", required=True)
    tokenizer_stats.add_argument("--special-token", action="append", default=[])
    tokenizer_stats.add_argument("--output")

    train_parser = subparsers.add_parser("train", help="train a Transformer language model")
    train_parser.add_argument("--train-data", required=True)
    train_parser.add_argument("--validation-data", required=True)
    train_parser.add_argument("--data-dtype", default="uint16")
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--device", default="cpu")
    train_parser.add_argument("--resume-from")
    train_parser.add_argument("--compile", action="store_true")
    _add_dataclass_arguments(train_parser, ModelConfig, "model-")
    _add_dataclass_arguments(train_parser, TrainConfig, "train-")

    decode = subparsers.add_parser("generate", help="generate text from a checkpoint")
    decode.add_argument("--checkpoint", required=True)
    decode.add_argument("--config", required=True)
    decode.add_argument("--vocab", required=True)
    decode.add_argument("--merges", required=True)
    decode.add_argument("--special-token", action="append", default=[])
    decode.add_argument("--prompt", required=True)
    decode.add_argument("--max-new-tokens", type=int, default=256)
    decode.add_argument("--temperature", type=float, default=0.8)
    decode.add_argument("--top-p", type=float, default=0.9)
    decode.add_argument("--device", default="cpu")
    decode.add_argument("--seed", type=int, default=42)

    plot = subparsers.add_parser("plot", help="render one or more metrics CSV files as an SVG loss curve")
    plot.add_argument("--metrics", action="append", required=True)
    plot.add_argument("--output", required=True)
    plot.add_argument("--split", choices=["train", "validation"], default="validation")
    plot.add_argument("--x-axis", choices=["iteration", "elapsed_seconds"], default="iteration")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "train-bpe":
        started_at = time.perf_counter()
        vocab, merges = train_bpe(
            args.input,
            args.vocab_size,
            args.special_token,
            num_processes=args.num_processes,
        )
        elapsed = time.perf_counter() - started_at
        for destination, value in ((args.vocab_out, vocab), (args.merges_out, merges)):
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "wb") as output_file:
                pickle.dump(value, output_file)
        longest_token: bytes = max(vocab.values(), key=lambda token: len(token))
        special_token_bytes = {token.encode("utf-8") for token in args.special_token}
        ordinary_tokens = [token for token in vocab.values() if token not in special_token_bytes]
        longest_ordinary_token: bytes = max(ordinary_tokens, key=lambda token: len(token))
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_bytes = peak_rss if sys.platform == "darwin" else peak_rss * 1024
        stats = {
            "elapsed_seconds": elapsed,
            "peak_rss_bytes": peak_rss_bytes,
            "vocab_size": len(vocab),
            "num_merges": len(merges),
            "longest_token_num_bytes": len(longest_token),
            "longest_token_hex": longest_token.hex(),
            "longest_token_text": longest_token.decode("utf-8", errors="backslashreplace"),
            "longest_non_special_token_num_bytes": len(longest_ordinary_token),
            "longest_non_special_token_hex": longest_ordinary_token.hex(),
            "longest_non_special_token_text": longest_ordinary_token.decode(
                "utf-8", errors="backslashreplace"
            ),
        }
        _write_or_print_json(stats, args.stats_out)
        return

    if args.command == "tokenize":
        count = tokenize_file(_load_tokenizer(args), args.input, args.output, dtype=args.dtype)
        print(f"wrote {count:,} tokens to {args.output}")
        return

    if args.command == "tokenizer-stats":
        tokenizer = _load_tokenizer(args)
        num_bytes = 0

        def counted_chunks():
            nonlocal num_bytes
            with open(args.input, encoding="utf-8") as input_file:
                for chunk in input_file:
                    num_bytes += len(chunk.encode("utf-8"))
                    yield chunk

        num_tokens = sum(1 for _ in tokenizer.encode_iterable(counted_chunks()))
        stats = {
            "input": args.input,
            "num_utf8_bytes": num_bytes,
            "num_tokens": num_tokens,
            "bytes_per_token": num_bytes / num_tokens if num_tokens else None,
            "compression_ratio": num_bytes / num_tokens if num_tokens else None,
        }
        _write_or_print_json(stats, args.output)
        return

    if args.command == "train":
        model_config = _dataclass_from_args(ModelConfig, args, "model_")
        train_config = _dataclass_from_args(TrainConfig, args, "train_")
        train_from_files(
            args.train_data,
            args.validation_data,
            data_dtype=args.data_dtype,
            model_config=model_config,
            train_config=train_config,
            output_dir=args.output_dir,
            device=args.device,
            resume_from=args.resume_from,
            compile_model=args.compile,
        )
        return

    if args.command == "generate":
        torch.manual_seed(args.seed)
        tokenizer = _load_tokenizer(args)
        with open(args.config, encoding="utf-8") as config_file:
            model_config = ModelConfig(**json.load(config_file)["model"])
        model = TransformerLM(**vars(model_config), device=torch.device(args.device))
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        prompt_tokens = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long, device=args.device)
        eos_id = tokenizer.special_token_to_id.get("<|endoftext|>")
        output = generate(
            model,
            prompt_tokens,
            args.max_new_tokens,
            context_length=model_config.context_length,
            temperature=args.temperature,
            top_p=args.top_p,
            eos_token_id=eos_id,
        )
        print(tokenizer.decode(output.tolist()))
        return

    if args.command == "plot":
        plot_metrics_svg(args.metrics, args.output, split=args.split, x_axis=args.x_axis)
        print(f"wrote loss curve to {args.output}")


if __name__ == "__main__":
    main()
