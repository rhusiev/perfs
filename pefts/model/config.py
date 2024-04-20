from dataclasses import dataclass

from ..peft import Peft


@dataclass
class Config:
    """The configuration for a Transformer model.

    Attributes:
        n_layers: The number of transformer layers.
        n_heads: The number of attention heads.
        d_model: The size of the hidden dimension.
        vocab_size: The size of the vocabulary.
        block_size: The size of the input block.
    """

    n_layers: int
    n_heads: int
    d_model: int
    vocab_size: int
    block_size: int

    peft: Peft | None = None

    @property
    def d_embedding(self) -> int:
        """The size of the key/query embedding dimension
        (d_k in the Attention is All You Need paper)."""
        return self.d_model // self.n_heads
