# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_assignment1_basics.pdf](./cs336_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment
We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv#installation) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using
```sh
uv run <python_file_path>
```
and the environment will be automatically solved and activated when necessary.

### Run unit tests


```sh
uv run pytest
```

The implementation in this repository is connected through
[tests/adapters.py](./tests/adapters.py). The full suite covers tokenizer training and encoding,
the Transformer, optimizer utilities, checkpointing, and a small end-to-end training pipeline.

### Download data
Download the TinyStories data and a subsample of OpenWebText

``` sh
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz

cd ..
```

## End-to-end workflow

All commands below run through the project CLI. Paths are examples and can be changed.

### 1. Train the TinyStories tokenizer

```sh
uv run python -m cs336_basics.cli train-bpe \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --vocab-size 10000 \
  --special-token '<|endoftext|>' \
  --vocab-out artifacts/tinystories_vocab.pkl \
  --merges-out artifacts/tinystories_merges.pkl \
  --stats-out artifacts/tinystories_bpe_stats.json
```

For large corpora, BPE pre-tokenization automatically uses up to eight CPU processes. The stats JSON
records elapsed time, process peak RSS, merge count, and the longest learned token.

### 2. Tokenize the training and validation corpora

The output is a raw `uint16` array plus a JSON metadata sidecar. The training code memory-maps
these files instead of loading the full tokenized corpus into RAM.

```sh
uv run python -m cs336_basics.cli tokenize \
  --input data/TinyStoriesV2-GPT4-train.txt \
  --output data/tinystories_train.bin \
  --vocab artifacts/tinystories_vocab.pkl \
  --merges artifacts/tinystories_merges.pkl \
  --special-token '<|endoftext|>'

uv run python -m cs336_basics.cli tokenize \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --output data/tinystories_valid.bin \
  --vocab artifacts/tinystories_vocab.pkl \
  --merges artifacts/tinystories_merges.pkl \
  --special-token '<|endoftext|>'
```

Measure the tokenizer's compression ratio on any corpus with:

```sh
uv run python -m cs336_basics.cli tokenizer-stats \
  --input data/TinyStoriesV2-GPT4-valid.txt \
  --vocab artifacts/tinystories_vocab.pkl \
  --merges artifacts/tinystories_merges.pkl \
  --special-token '<|endoftext|>' \
  --output artifacts/tinystories_compression.json
```

### 3. Train on CPU

The defaults use the assignment's 17M-parameter architecture and low-resource training budget:
batch size 32, context length 256, and 5,000 steps (about 41M tokens). Metrics are written to
`metrics.csv`; the configuration and latest checkpoint are saved in the run directory.

```sh
uv run python -m cs336_basics.cli train \
  --train-data data/tinystories_train.bin \
  --validation-data data/tinystories_valid.bin \
  --output-dir runs/tinystories_cpu \
  --device cpu \
  --compile
```

Resume an interrupted run by passing the existing checkpoint. Keep the model arguments identical and
set `--train-max-steps` to the desired final step.

```sh
uv run python -m cs336_basics.cli train \
  --train-data data/tinystories_train.bin \
  --validation-data data/tinystories_valid.bin \
  --output-dir runs/tinystories_cpu \
  --device cpu \
  --resume-from runs/tinystories_cpu/checkpoint.pt
```

For quick debugging, reduce `--model-d-model`, `--model-num-layers`,
`--model-context-length`, and `--train-max-steps`.

The architecture ablations from the handout are available as model arguments:

| Experiment | Arguments |
|---|---|
| Remove RMSNorm | `--model-norm-style none` |
| Post-norm | `--model-norm-style post` |
| NoPE | `--model-use-rope false` |
| SiLU FFN | `--model-ffn-type silu --model-d-ff 2048` |

Use a separate output directory for every run so their checkpoints and loss curves remain comparable.

### 4. Generate text

```sh
uv run python -m cs336_basics.cli generate \
  --checkpoint runs/tinystories_cpu/checkpoint.pt \
  --config runs/tinystories_cpu/config.json \
  --vocab artifacts/tinystories_vocab.pkl \
  --merges artifacts/tinystories_merges.pkl \
  --special-token '<|endoftext|>' \
  --prompt 'Once upon a time' \
  --max-new-tokens 256 \
  --temperature 0.8 \
  --top-p 0.9
```

Run `uv run python -m cs336_basics.cli <command> --help` for every configurable option.

### 5. Plot learning curves

Pass `--metrics` more than once to compare runs. Use `--x-axis elapsed_seconds` for the leaderboard's
wall-clock plot.

```sh
uv run python -m cs336_basics.cli plot \
  --metrics runs/tinystories_cpu/metrics.csv \
  --metrics runs/tinystories_nope/metrics.csv \
  --output runs/tinystories_comparison.svg \
  --split validation \
  --x-axis iteration
```
