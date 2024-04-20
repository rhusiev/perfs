import torch
import torch.nn as nn

from ..peft import Peft


class LoRALinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, lora_rank=4):
        super().__init__(in_features, out_features, bias=bias)
        self.lora_rank = lora_rank
        self.peft_lora_A = nn.Parameter(torch.zeros(lora_rank, out_features))
        self.peft_lora_B = nn.Parameter(torch.zeros(in_features, lora_rank))
        nn.init.normal_(self.peft_lora_A, std=1.0 / lora_rank)
        nn.init.zeros_(self.peft_lora_B)

    def forward(self, x):
        # Apply LoRA
        weight = self.weight + self.peft_lora_B @ self.peft_lora_A
        # Apply Linear layer
        return nn.functional.linear(x, weight, self.bias)


class LoRAPeft(Peft):
    def __init__(self, lora_rank=4) -> None:
        self.lora_rank = lora_rank
        self.layers: list[LoRALinear] = []

    def __call__(self, in_features: int, out_features: int) -> nn.Linear:
        layer = LoRALinear(in_features, out_features, lora_rank=self.lora_rank)
        self.layers.append(layer)
        return layer

    def state_dict(self) -> dict[str, nn.Parameter]:
        return {
            f"layers.{i}.peft_lora_A": layer.peft_lora_A
            for i, layer in enumerate(self.layers)
        } | {
            f"layers.{i}.peft_lora_B": layer.peft_lora_B
            for i, layer in enumerate(self.layers)
        }

    def load_state_dict(self, state_dict: dict[str, nn.Parameter]) -> None:
        for i, layer in enumerate(self.layers):
            layer.peft_lora_A = state_dict[f"layers.{i}.peft_lora_A"]
            layer.peft_lora_B = state_dict[f"layers.{i}.peft_lora_B"]
