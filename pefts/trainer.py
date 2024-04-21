from typing import Iterable

import torch

from .model import Transformer


class Trainer:
    """A simple trainer for the transformer model."""

    def __init__(self, model: Transformer, optimizer: torch.optim.Optimizer) -> None:
        """Initialize the trainer.

        Args:
            model: The model to train.
            optimizer: The optimizer to use.
        """
        self.model = model
        self.optimizer = optimizer
        self.device = next(model.parameters()).device

    def train(self, dataset: Iterable[torch.Tensor], epochs: int = 1) -> None:
        """Train the model on the given dataset.

        Args:
            dataset: The dataset to train on.
            epochs: The number of epochs to train for.
        """
        for _ in range(epochs):
            for batch in dataset:
                self.optimizer.zero_grad()
                _, loss = self.model(batch[:, :-1].to(self.device), batch[:, 1:].to(self.device))
                print(loss.item())
                loss.backward()
                self.optimizer.step()
