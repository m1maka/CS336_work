from __future__ import annotations

import os
import re
import heapq
import multiprocessing as mp
from collections import Counter, defaultdict
from collections.abc import Iterable

from .tokenizer import GPT2_PRETOKENIZE_PATTERN


class _ReversePair:
    """Reverse byte-pair ordering so heap ties follow the assignment rule."""

    __slots__ = ("pair",)

    def __init__(self, pair: tuple[bytes, bytes]) -> None:
        self.pair = pair

    def __lt__(self, other: _ReversePair) -> bool:
        return self.pair > other.pair


def _split_on_special_tokens(text: str, special_tokens: list[str]) -> Iterable[str]:
    """Yield ordinary-text spans, treating special tokens as hard boundaries."""
    if not special_tokens:
        yield text
        return

    ordered_tokens: list[str] = sorted(set(special_tokens), key=lambda token: len(token), reverse=True)
    pattern = "|".join(re.escape(token) for token in ordered_tokens)
    yield from re.split(pattern, text)


def _count_text(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    counts: Counter[tuple[bytes, ...]] = Counter()
    for ordinary_text in _split_on_special_tokens(text, special_tokens):
        for match in GPT2_PRETOKENIZE_PATTERN.finditer(ordinary_text):
            counts[tuple(bytes([byte]) for byte in match.group().encode("utf-8"))] += 1
    return counts


def _find_chunk_boundaries(
    input_path: str | os.PathLike[str], num_chunks: int, delimiter: bytes
) -> list[int]:
    """Find byte offsets at delimiter starts near evenly spaced file positions."""
    file_size = os.path.getsize(input_path)
    if num_chunks <= 1 or not delimiter:
        return [0, file_size]

    boundaries = [0]
    with open(input_path, "rb") as input_file:
        for chunk_index in range(1, num_chunks):
            cursor = file_size * chunk_index // num_chunks
            input_file.seek(cursor)
            tail = b""
            boundary = file_size
            while cursor < file_size:
                block = input_file.read(4096)
                if not block:
                    break
                searchable = tail + block
                match_index = searchable.find(delimiter)
                if match_index >= 0:
                    boundary = cursor - len(tail) + match_index
                    break
                tail = searchable[-(len(delimiter) - 1) :] if len(delimiter) > 1 else b""
                cursor += len(block)
            if boundaries[-1] < boundary < file_size:
                boundaries.append(boundary)
    boundaries.append(file_size)
    return boundaries


def _count_file_region(
    arguments: tuple[str, int, int, list[str]],
) -> Counter[tuple[bytes, ...]]:
    input_path, start, end, special_tokens = arguments
    with open(input_path, "rb") as input_file:
        input_file.seek(start)
        text = input_file.read(end - start).decode("utf-8")
    return _count_text(text, special_tokens)


class BPETrainer:
    """Train a byte-level BPE vocabulary with deterministic merge ordering."""

    def __init__(self, vocab_size: int, special_tokens: list[str] | None = None) -> None:
        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        minimum_size = 256 + len(self.special_tokens)
        if vocab_size < minimum_size:
            raise ValueError(f"vocab_size must be at least {minimum_size}")
        if "" in self.special_tokens:
            raise ValueError("special tokens cannot contain the empty string")
        self.vocab_size = vocab_size

    def _count_pretokens(self, texts: Iterable[str]) -> Counter[tuple[bytes, ...]]:
        counts: Counter[tuple[bytes, ...]] = Counter()
        for text in texts:
            counts.update(_count_text(text, self.special_tokens))
        return counts

    def train(self, texts: Iterable[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        return self.train_counts(self._count_pretokens(texts))

    def train_counts(
        self, word_counts: Counter[tuple[bytes, ...]]
    ) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        words = [list(word) for word in word_counts]
        frequencies = list(word_counts.values())

        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        pair_to_words: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)
        for word_index, symbols in enumerate(words):
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] += frequencies[word_index]
                pair_to_words[pair].add(word_index)

        pair_heap = [(-count, _ReversePair(pair), pair) for pair, count in pair_counts.items()]
        heapq.heapify(pair_heap)

        vocab: dict[int, bytes] = {
            token_id: token.encode("utf-8") for token_id, token in enumerate(self.special_tokens)
        }
        next_id = len(vocab)
        for byte in range(256):
            vocab[next_id] = bytes([byte])
            next_id += 1

        merges: list[tuple[bytes, bytes]] = []
        while len(vocab) < self.vocab_size and pair_counts:
            best_pair = None
            while pair_heap:
                negative_count, _, candidate = heapq.heappop(pair_heap)
                if pair_counts.get(candidate) == -negative_count:
                    best_pair = candidate
                    break
            if best_pair is None or pair_counts[best_pair] <= 0:
                break

            vocab[len(vocab)] = best_pair[0] + best_pair[1]
            merges.append(best_pair)

            affected_words = list(pair_to_words.get(best_pair, ()))
            changed_pairs: set[tuple[bytes, bytes]] = set()
            for word_index in affected_words:
                old_symbols = words[word_index]
                frequency = frequencies[word_index]
                old_pairs = set(zip(old_symbols, old_symbols[1:]))
                for pair in zip(old_symbols, old_symbols[1:]):
                    pair_counts[pair] -= frequency
                    changed_pairs.add(pair)

                merged_symbols: list[bytes] = []
                index = 0
                while index < len(old_symbols):
                    if index + 1 < len(old_symbols) and (old_symbols[index], old_symbols[index + 1]) == best_pair:
                        merged_symbols.append(old_symbols[index] + old_symbols[index + 1])
                        index += 2
                    else:
                        merged_symbols.append(old_symbols[index])
                        index += 1
                words[word_index] = merged_symbols

                new_pairs = set(zip(merged_symbols, merged_symbols[1:]))
                for pair in old_pairs - new_pairs:
                    pair_to_words[pair].discard(word_index)
                for pair in zip(merged_symbols, merged_symbols[1:]):
                    pair_counts[pair] += frequency
                    pair_to_words[pair].add(word_index)
                    changed_pairs.add(pair)

            pair_counts.pop(best_pair, None)
            pair_to_words.pop(best_pair, None)
            changed_pairs.discard(best_pair)
            for pair in changed_pairs:
                count = pair_counts[pair]
                if count > 0:
                    heapq.heappush(pair_heap, (-count, _ReversePair(pair), pair))
                else:
                    pair_counts.pop(pair, None)
                    pair_to_words.pop(pair, None)

        return vocab, merges


def train_bpe(
    input_path: str | os.PathLike[str],
    vocab_size: int,
    special_tokens: list[str] | None = None,
    num_processes: int | None = None,
    **_: object,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train BPE from a UTF-8 text file."""
    trainer = BPETrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    file_size = os.path.getsize(input_path)
    if num_processes is None:
        num_processes = min(os.cpu_count() or 1, 8) if file_size >= 50_000_000 and trainer.special_tokens else 1
    if num_processes <= 0:
        raise ValueError("num_processes must be positive")

    if num_processes > 1 and trainer.special_tokens:
        delimiter = trainer.special_tokens[0].encode("utf-8")
        boundaries = _find_chunk_boundaries(input_path, num_processes * 4, delimiter)
        arguments = [
            (os.fspath(input_path), start, end, trainer.special_tokens)
            for start, end in zip(boundaries, boundaries[1:])
        ]
        word_counts: Counter[tuple[bytes, ...]] = Counter()
        with mp.Pool(processes=num_processes) as pool:
            for partial_counts in pool.imap_unordered(_count_file_region, arguments):
                word_counts.update(partial_counts)
        return trainer.train_counts(word_counts)

    with open(input_path, encoding="utf-8") as input_file:
        # Pretokenization is sensitive to chunk boundaries (notably runs of
        # whitespace), so a file's lines must not be treated as documents.
        return trainer.train([input_file.read()])
