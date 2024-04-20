import torch
import torch.nn as nn
from transformers.activations import NewGELUActivation

from .config import Config


class FeedForward(nn.Module):
    """A feedforward layer for the transformer."""

    def __init__(self, config: Config) -> None:
        """Initialize the layer.

        Args:
            config: The configuration for the model.
        """
        super().__init__()

        self.upprojection = nn.Linear(config.d_model, 4 * config.d_model)
        self.activation = NewGELUActivation()
        self.downprojection = nn.Linear(4 * config.d_model, config.d_model)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass through the
        feedforward layer.

        Args:
            e: The input embedding, with shape (batch_size, seq_len, d_model).

        Returns:
            The output embedding, with shape (batch_size, seq_len, d_model).
        """
        return self.downprojection(self.activation(self.upprojection(e)))
