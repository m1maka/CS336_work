from __future__ import annotations

import torch
from torch import Tensor, nn


def sample_next_token(logits: Tensor, temperature: float = 1.0, top_p: float = 1.0) -> Tensor:
    """Sample one token per row using temperature and nucleus sampling."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must lie in (0, 1]")

    probabilities = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, dim=-1, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        # Keep the first token that takes the cumulative mass over top_p.
        remove = cumulative - sorted_probabilities >= top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
        sorted_probabilities /= sorted_probabilities.sum(dim=-1, keepdim=True)
        sampled_sorted_index = torch.multinomial(sorted_probabilities, num_samples=1)
        return sorted_indices.gather(dim=-1, index=sampled_sorted_index).squeeze(-1)
    return torch.multinomial(probabilities, num_samples=1).squeeze(-1)


@torch.no_grad()
def generate(
    model: nn.Module,
    prompt_tokens: Tensor,
    max_new_tokens: int,
    *,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    eos_token_id: int | None = None,
) -> Tensor:
    """Autoregressively extend a one-dimensional token prompt."""
    if prompt_tokens.ndim != 1 or prompt_tokens.numel() == 0:
        raise ValueError("prompt_tokens must be a non-empty one-dimensional tensor")
    if max_new_tokens < 0 or context_length <= 0:
        raise ValueError("max_new_tokens must be non-negative and context_length must be positive")

    was_training = model.training
    model.eval()
    generated = prompt_tokens
    for _ in range(max_new_tokens):
        model_input = generated[-context_length:].unsqueeze(0)
        next_logits = model(model_input)[0, -1]
        next_token = sample_next_token(next_logits, temperature=temperature, top_p=top_p)
        generated = torch.cat((generated, next_token.reshape(1)))
        if eos_token_id is not None and next_token.item() == eos_token_id:
            break
    model.train(was_training)
    return generated
