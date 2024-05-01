import torch
import torch.nn as nn

from ..peft import Peft


class LoRALinear(nn.Linear):
    """A single Linear layer with LoRA (Low-Rank Adaptation)"""

    def __init__(self, in_features, out_features, bias=True, lora_rank=4):
        """Initialize a Linear layer with LoRA

        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            bias (bool, optional): Whether to include bias. Defaults to True.
            lora_rank (int, optional): Rank of the LoRA matrix. Defaults to 4.
        """
        super().__init__(in_features, out_features, bias=bias)
        self.lora_rank = lora_rank
        self.peft_lora_A = nn.Parameter(torch.zeros(lora_rank, out_features))
        self.peft_lora_B = nn.Parameter(torch.zeros(in_features, lora_rank))
        nn.init.normal_(self.peft_lora_A, std=1.0 / lora_rank)
        nn.init.zeros_(self.peft_lora_B)

        self.enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Linear layer with LoRA

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        if not self.enabled:
            return nn.functional.linear(x, self.weight, self.bias)
        # Apply LoRA
        weight = self.weight + self.peft_lora_B @ self.peft_lora_A
        # Apply Linear layer
        return nn.functional.linear(x, weight, self.bias)

    def to(self, *args, **kwargs) -> "LoRALinear":
        """Move the Linear layer with LoRA to a device

        Returns:
            LoRALinear: Linear layer with LoRA on a device
        """
        self.peft_lora_A = self.peft_lora_A.to(*args, **kwargs)
        self.peft_lora_B = self.peft_lora_B.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class LoRAPeft(Peft):
    """LoRA (Low-Rank Adaptation) for Linear layers"""

    def __init__(self, lora_rank=4) -> None:
        """Initialize LoRA for Linear layers

        Args:
            lora_rank (int, optional): Rank of the LoRA matrix. Defaults to 4.
        """
        self.lora_rank = lora_rank
        self.layers: list[LoRALinear] = []

    def __call__(self, in_features: int, out_features: int) -> nn.Linear:
        """Create a Linear layer with LoRA and add it to the list of layers

        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features

        Returns:
            nn.Linear: Linear layer with LoRA
        """
        layer = LoRALinear(in_features, out_features, lora_rank=self.lora_rank)
        self.layers.append(layer)
        return layer

    def state_dict(self) -> dict[str, nn.Parameter]:
        """Get the state dictionary of the LoRA layers for saving
        You can use torch.save(peft.state_dict(), "finetune.pth")

        Returns:
            dict[str, nn.Parameter]: State dictionary of the LoRA layers
        """
        return {
            f"layers.{i}.peft_lora_A": layer.peft_lora_A
            for i, layer in enumerate(self.layers)
        } | {
            f"layers.{i}.peft_lora_B": layer.peft_lora_B
            for i, layer in enumerate(self.layers)
        }

    def load_state_dict(self, state_dict: dict[str, nn.Parameter]) -> None:
        """Load the state dictionary of the LoRA layers

        Args:
            state_dict (dict[str, nn.Parameter]): State dictionary of the LoRA layers
        """
        for i, layer in enumerate(self.layers):
            layer.peft_lora_A = state_dict[f"layers.{i}.peft_lora_A"]
            layer.peft_lora_B = state_dict[f"layers.{i}.peft_lora_B"]

    def enable_all(self) -> None:
        """Enable all LoRA layers"""
        for layer in self.layers:
            layer.enabled = True

    def disable_all(self) -> None:
        """Disable all LoRA layers"""
        for layer in self.layers:
            layer.enabled = False
