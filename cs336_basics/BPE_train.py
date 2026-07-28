from unittest import result

from tokenizer import GPT2_PRETOKENIZE_PATTERN

from datasets import load_dataset

# 加载数据集，可以指定 'train' 或 'validation' 分片
dataset = load_dataset("roneneldan/TinyStories", split="train")

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
    
    #增加迭代训练的功能
    def iterative_train(self, texts: list[str], iterations: int) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
        vocab: dict[int, bytes] = {i: token.encode("utf-8") for i, token in enumerate(self.special_tokens)}
        for _ in range(iterations):
            pair_freqs: dict[tuple[bytes, bytes], int] = {}
            for text in texts:
                for pretoken in GPT2_PRETOKENIZE_PATTERN.findall(text):
                    token_bytes = pretoken.encode("utf-8")
                    for i in range(len(token_bytes) - 1):
                        pair = (token_bytes[i : i + 1], token_bytes[i + 1 : i + 2])
                        pair_freqs[pair] = pair_freqs.get(pair, 0) + 1

            most_common_pairs = sorted(pair_freqs.items(), key=lambda item: item[1], reverse=True)
            merges = [pair for pair, freq in most_common_pairs[: self.vocab_size - len(vocab)]]

            for first_byte, second_byte in merges:
                new_token = first_byte + second_byte
                if new_token not in vocab.values():
                    vocab[len(vocab)] = new_token

            #更新文本，将合并的字节对替换为新的token
            result = []
            for text in texts:
                for first_byte, second_byte in merges:
                    text =text.replace(first_byte.decode("utf-8") + second_byte.decode("utf-8"), 
                                      (first_byte + second_byte).decode("utf-8"))
                result.append(text)
            texts = result

        return vocab, merges

# Example usage:
trainer = BPETrainer(vocab_size=1000, special_tokens=["<PAD>", "<UNK>"])
texts = [item["text"] for item in dataset]
vocab, merges = trainer.iterative_train(texts, iterations=10)
print("Vocabulary:", vocab)
print("Merges:", merges)