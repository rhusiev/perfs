from typing import Protocol

import torch.nn as nn


class Peft(Protocol):
    """A protocol for parameter-efficient fine-tuning modules."""

    def __call__(self, in_features: int, out_features: int) -> nn.Linear:
        """Create a Linear layer with Peft.

        Args:
            in_features: The number of input features.
            out_features: The number of output features.

        Returns:
            A Linear layer with Peft, as a subclass of nn.Linear.
        """
        ...

    def state_dict(self) -> dict[str, nn.Parameter]:
        """Return the state dictionary of the Peft module.
        You can use torch.save(peft.state_dict(), "finetune.pth").

        Returns:
            The state dictionary of the Peft module.
        """
        ...

    def load_state_dict(self, state_dict: dict[str, nn.Parameter]) -> None:
        """Load the state dictionary of the Peft module.

        Args:
            state_dict: The state dictionary of the Peft module.
        """
        ...
