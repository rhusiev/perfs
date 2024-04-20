import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .config import Config
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """A single simple transformer block with a multi-head attention"""

    def __init__(self, config: Config) -> None:
        """Initialize the block.

        Args:
            config: The configuration for the model.
        """
        super().__init__()

        # Keeping some of their names so that GPT2 weights can be loaded
        # more easily
        self.ln_1 = nn.LayerNorm(config.d_model)
        self.ln_2 = nn.LayerNorm(config.d_model)

        self.attn = MultiHeadAttention(config)
        # AKA mlp, multi-layer perceptron:
        self.feedforward = FeedForward(config)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass through the
        transformer block.

        Args:
            e: The input embedding, with shape (batch_size, seq_len, d_model).

        Returns:
            The output embedding, with shape (batch_size, seq_len, d_model).
        """
        # I wish I could just normalize the input *after* the attention
        # and feedforward layers, to match the Attention is All You Need
        # paper, but the pretrained GPT-2 model does it before, so I'm
        # doing it here as well. I tried changing it around, but it become
        # way too complicated (cause remember - you skip the normalization)
        # even before I could get it working.

        # Note: we can't use += - PyTorch doesn't support it, saving gradients
        # breaks it. So we have to do it like this.
        e = e + self.attn(self.ln_1(e))
        e = e + self.feedforward(self.ln_2(e))
        return e
