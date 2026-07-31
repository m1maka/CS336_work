from __future__ import annotations

import pickle
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from os import PathLike

import regex as re

# GPT-2 pre-tokenization regex.
GPT2_PRETOKENIZE_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    _MAX_CACHE_SIZE = 4096

    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.id_to_token: dict[int, bytes] = dict(vocab)
        self.token_to_id: dict[bytes, int] = {token_bytes: token_id for token_id, token_bytes in self.id_to_token.items()}
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {pair: rank for rank, pair in enumerate(merges)}
        self.cache: OrderedDict[bytes, tuple[int, ...]] = OrderedDict()

        self.special_tokens = list(dict.fromkeys(special_tokens or []))
        if "" in self.special_tokens:
            raise ValueError("A special token cannot be the empty string.")
        self.special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        self.special_token_to_id: dict[str, int] = {}
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                token_id = max(self.id_to_token, default=-1) + 1
                self.id_to_token[token_id] = token_bytes
                self.token_to_id[token_bytes] = token_id
            self.special_token_to_id[special_token] = self.token_to_id[token_bytes]

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | PathLike[str],
        merges_filepath: str | PathLike[str],
        special_tokens: list[str] | None = None,
    ) -> BPETokenizer:
        """Construct a tokenizer from a vocab and merges serialized with pickle."""
        with open(vocab_filepath, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)
        with open(merges_filepath, "rb") as merges_file:
            merges = pickle.load(merges_file)
        if not isinstance(vocab, dict) or not isinstance(merges, list):
            raise ValueError("Expected a pickled vocab dict and a pickled merges list.")
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for is_special, segment in self._split_by_special_tokens(text):
            if is_special:
                token_ids.append(self.special_token_to_id[segment])
            else:
                token_ids.extend(self._encode_non_special_text(segment))
        return token_ids

    def decode(self, token_ids: list[int]) -> str:
        token_bytes = b"".join(self.id_to_token[token_id] for token_id in token_ids)
        return token_bytes.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        pending = ""
        for chunk in iterable:
            if not isinstance(chunk, str):
                raise TypeError("encode_iterable expects an iterable of strings.")
            pending += chunk
            safe_end = self._streaming_safe_end(pending)
            if safe_end:
                yield from self.encode(pending[:safe_end])
                pending = pending[safe_end:]
        yield from self.encode(pending)

    def _streaming_safe_end(self, text: str) -> int:
        """Find a boundary whose tokenization cannot depend on the next chunk."""
        if not text:
            return 0

        matches = list(GPT2_PRETOKENIZE_PATTERN.finditer(text))
        if not matches:
            return 0

        # The final pre-token may continue in the next chunk.
        safe_end = matches[-1].start()

        # Retain enough text to recognize a special token split across chunks.
        if self.special_tokens_sorted:
            boundary = max(0, len(text) - len(self.special_tokens_sorted[0]) + 1)
            for match in matches:
                if match.start() < boundary < match.end():
                    boundary = match.start()
                    break
            safe_end = min(safe_end, boundary)

            # Do not split through an already-complete special token.
            for special_token in self.special_tokens_sorted:
                search_start = 0
                while True:
                    token_start = text.find(special_token, search_start)
                    if token_start < 0:
                        break
                    token_end = token_start + len(special_token)
                    if token_start < safe_end < token_end:
                        safe_end = token_start
                    search_start = token_start + 1

        return safe_end

    def _encode_non_special_text(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for pretoken in GPT2_PRETOKENIZE_PATTERN.findall(text):
            token_ids.extend(self._encode_bytes_with_bpe(pretoken.encode("utf-8")))
        return token_ids

    def _encode_bytes_with_bpe(self, token_bytes: bytes) -> list[int]:
        if not token_bytes:
            return []
        cached = self.cache.get(token_bytes)
        if cached is not None:
            # Keep frequently reused byte sequences hot in the LRU cache.
            self.cache.move_to_end(token_bytes)
            return list(cached)

        symbols = [bytes([b]) for b in token_bytes]
        while len(symbols) > 1:
            best_pair = None
            best_rank = None
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                rank = self.merge_ranks.get(pair)
                if rank is None:
                    continue
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_pair = pair

            if best_pair is None:
                break

            merged: list[bytes] = []
            i = 0
            while i < len(symbols):
                if i < len(symbols) - 1 and symbols[i] == best_pair[0] and symbols[i + 1] == best_pair[1]:
                    merged.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            symbols = merged
        
        token_ids = [self.token_to_id[symbol] for symbol in symbols]
        self.cache[token_bytes] = tuple(token_ids)
        self.cache.move_to_end(token_bytes)
        if len(self.cache) > self._MAX_CACHE_SIZE:
            self.cache.popitem(last=False)
        return token_ids

    def _split_by_special_tokens(self, text: str) -> list[tuple[bool, str]]:
        if not self.special_tokens_sorted:
            return [(False, text)]

        out: list[tuple[bool, str]] = []
        i = 0
        start = 0
        n = len(text)

        while i < n:
            matched_special = None
            for special_token in self.special_tokens_sorted:
                if text.startswith(special_token, i):
                    matched_special = special_token
                    break

            if matched_special is None:
                i += 1
                continue

            if start < i:
                out.append((False, text[start:i]))
            out.append((True, matched_special))

            i += len(matched_special)
            start = i

        if start < n:
            out.append((False, text[start:]))
        return out


def get_tokenizer(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> BPETokenizer:
    return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special_tokens)
