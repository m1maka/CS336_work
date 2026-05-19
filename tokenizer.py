from __future__ import annotations

from collections.abc import Iterable, Iterator

import regex as re

# GPT-2 pre-tokenization regex.
GPT2_PRETOKENIZE_PATTERN = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.id_to_token: dict[int, bytes] = dict(vocab)
        self.token_to_id: dict[bytes, int] = {token_bytes: token_id for token_id, token_bytes in self.id_to_token.items()}
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {pair: rank for rank, pair in enumerate(merges)}
        self.cache: dict[bytes, list[int]] = {}

        self.special_tokens = special_tokens or []
        self.special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
        self.special_token_to_id: dict[str, int] = {}
        for special_token in self.special_tokens:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.token_to_id:
                raise ValueError(f"Special token {special_token!r} does not exist in vocab.")
            self.special_token_to_id[special_token] = self.token_to_id[token_bytes]

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
        return token_bytes.decode("utf-8", errors="strict")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # Stream over incoming chunks to keep memory usage low.
        for chunk in iterable:
            for token_id in self.encode(chunk):
                yield token_id

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
            return cached

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
        self.cache[token_bytes] = token_ids
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
