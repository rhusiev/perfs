import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class MultiHeadAttention(nn.Module):
    """A multi-head attention layer."""

    def __init__(self, config: Config) -> None:
        """Initialize the layer.

        Args:
            config: The configuration for the model.
        """
        super().__init__()

        # key, query, value projections for all heads, but as one concatenated tensor
        # We're keeping them conctenated just so that we can do LoRA finetunes
        # on them - otherwise we would split them up for readability
        # self.q = nn.Linear(config.d_model, config.d_model)
        self.q = (config.peft or nn.Linear)(config.d_model, config.d_model)
        self.k = nn.Linear(config.d_model, config.d_model)
        self.v = (config.peft or nn.Linear)(config.d_model, config.d_model)

        self.output = nn.Linear(config.d_model, config.d_model)

        self.config = config

    def _split_heads(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Split the key, query, and value tensors into multiple heads.

        Args:
            e: The key, query, or value tensor,
                with shape (batch_size, seq_len, d_model).
            seq_len: The length of the input sequence.

        Returns:
            The key, query, or value tensor with an additional head dimension,
            with shape (batch_size, n_heads, seq_len, d_embedding).
        """
        return x.view(
            -1, seq_len, self.config.n_heads, self.config.d_embedding
        ).transpose(1, 2)

    def _mask_out_future(self, att: torch.Tensor) -> torch.Tensor:
        """Mask out the future in the attention weights.
        This is the triangle thingy for the "causal LM" part.

        Args:
            att: The attention weights,
                with shape (batch_size, n_heads, seq_len, seq_len).

        Returns:
            The attention weights with the future masked out.
        """
        return att.masked_fill(
            torch.tril(torch.ones(att.size(-2), att.size(-1))).view(
                1, 1, att.size(-2), att.size(-1)
            ).to(att.device)
            == 0,
            float("-inf"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass through the
        multi-head attention layer.

        Args:
            x: The input embedding, with shape (batch_size, seq_len, d_model).

        Returns:
            The output embedding, with shape (batch_size, seq_len, d_model).
        """
        seq_len = x.size(1)

        q = self._split_heads(self.q(x), seq_len)
        k = self._split_heads(self.k(x), seq_len)
        v = self._split_heads(self.v(x), seq_len)

        att = (q @ k.transpose(-2, -1)) * (self.config.d_embedding**-0.5)
        att = self._mask_out_future(att)
        att = F.softmax(att, dim=-1)

        y = (att @ v).transpose(1, 2).contiguous().view(*x.size())

        return self.output(y)
