import importlib.metadata

from .attention import MultiHeadSelfAttention, scaled_dot_product_attention, softmax
from .embedding import Embedding
from .linear import Linear
from .rmsnorm import RMSNorm
from .rope import RotaryPositionalEmbedding
from .swiglu import SwiGLU, silu
from .transformer_block import TransformerBlock
from .transformer_lm import TransformerLM

try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass

__all__ = [
    "Embedding",
    "Linear",
    "MultiHeadSelfAttention",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SwiGLU",
    "TransformerBlock",
    "TransformerLM",
    "scaled_dot_product_attention",
    "silu",
    "softmax",
]
