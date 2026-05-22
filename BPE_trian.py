from tokenizer import GPT2_PRETOKENIZE_PATTERN

class BPETrainer:
    def __init__(self, vocab_size: int, special_tokens: list[str] | None = None) -> None:
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or []
        self.special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)

    def train(self, texts: list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        # Count byte pair frequencies.
        pair_freqs: dict[tuple[bytes, bytes], int] = {}
        for text in texts:
            for pretoken in GPT2_PRETOKENIZE_PATTERN.findall(text):
                token_bytes = pretoken.encode("utf-8")
                for i in range(len(token_bytes) - 1):
                    pair = (token_bytes[i : i + 1], token_bytes[i + 1 : i + 2])
                    pair_freqs[pair] = pair_freqs.get(pair, 0) + 1

        # Sort pairs by frequency and take the most common ones.
        most_common_pairs = sorted(pair_freqs.items(), key=lambda item: item[1], reverse=True)
        merges = [pair for pair, freq in most_common_pairs[: self.vocab_size - len(self.special_tokens)]]

        # Build the final vocab.
        vocab: dict[int, bytes] = {}
        for i, special_token in enumerate(self.special_tokens):
            vocab[i] = special_token.encode("utf-8")
        for i, (first_byte, second_byte) in enumerate(merges, start=len(self.special_tokens)):
            vocab[i] = first_byte + second_byte

        return vocab, merges