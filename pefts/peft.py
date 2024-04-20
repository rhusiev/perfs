from typing import Protocol

import torch.nn as nn


class Peft(Protocol):
    def __call__(self, in_features: int, out_features: int) -> nn.Linear:
        ...

    def state_dict(self) -> dict[str, nn.Parameter]:
        ...

    def load_state_dict(self, state_dict: dict[str, nn.Parameter]) -> None:
        ...
