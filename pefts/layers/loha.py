import torch
import torch.nn as nn

from ..peft import Peft

class HadamardProductError(ValueError):
    """Error if sizes does not match
    """

    def __init__(self, message="For Hadamart product matrices should have the same size!!"):
        self.message = message
        super().__init__(self.message)

class LoHaLinear(nn.Linear):
    """A single Linear layer with LoHa (Low-Rank and Hadamard Product Adaptation)"""

    def __init__(self, in_features, out_features, bias=True, loha_rank=4):
        """Initialize a Linear layer with LoHa
        Bathed on the theory: B1A1 hadamart product B2A2

        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            bias (bool, optional): Whether to include bias. Defaults to True.
            rank (int, optional): Rank of the LoHa matrix. Defaults to 4.
        """
        super().__init__(in_features, out_features, bias=bias)
        self.loha_rank = loha_rank
        
        self.peft_loha_B1 = nn.Parameter(torch.zeros(in_features, loha_rank))
        self.peft_loha_A1 = nn.Parameter(torch.zeros(loha_rank, out_features))
        nn.init.normal_(self.peft_loha_A1, std=1.0 / loha_rank)
        nn.init.zeros_(self.peft_loha_B1)

        self.peft_loha_B2 = nn.Parameter(torch.zeros(in_features, loha_rank))
        self.peft_loha_A2 = nn.Parameter(torch.zeros(loha_rank, out_features))
        nn.init.normal_(self.peft_loha_A2, std=1.0 / loha_rank)
        nn.init.zeros_(self.peft_loha_B2)

        #for scaling parameter, single value with standard normal distribution
        self.peft_loha_gamma_value = nn.Parameter(torch.randn(1))
    
    def calculating_hadamard(self):
        """Calculating Hadamart Product

        Args:
            None
        Returns:
            HadamardProductSizeError: If the first matrix and the second one have different sizes.
            torch.Tensor: Hadamart product of the first matrix and the second one, stacked along the specified dimension.
        """
        first_matrix = self.peft_loha_B1.mm(self.peft_loha_A1)
        second_matrix = self.peft_loha_B2.mm(self.peft_loha_A2)
        
        if first_matrix.size() != second_matrix.size():
            raise HadamardProductError()
        else:
            size = first_matrix.size(0)
            return torch.stack([first_matrix[i] * second_matrix[i] for i in range(size)], dim=0)
        
    def built_in(self):
        """Built in product
        """
        first_matrix = self.peft_loha_B1.mm(self.peft_loha_A1)
        second_matrix = self.peft_loha_B2.mm(self.peft_loha_A2)
        return first_matrix.mul(second_matrix)    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Linear layer with LoHa

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Apply LoHa
        weight = self.weight + self.peft_loha_gamma_value * self.calculating_hadamard()
        # Apply Linear layer
        return nn.functional.linear(x, weight, self.bias)
    
    def to(self, *args, **kwargs):
        """Move the Linear layer with LoHa to a specified device

        Args:
            *args: Arguments for the to method
            **kwargs: Keyword arguments for the to method

        Returns:
            LoHaLinear: Linear layer with LoHa on the specified device
        """
        self.peft_loha_B1 = self.peft_loha_B1.to(*args, **kwargs)
        self.peft_loha_A1 = self.peft_loha_A1.to(*args, **kwargs)
        self.peft_loha_B2 = self.peft_loha_B2.to(*args, **kwargs)
        self.peft_loha_A2 = self.peft_loha_A2.to(*args, **kwargs)
        self.peft_loha_gamma_value = self.peft_loha_gamma_value.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class LoHaPeft(Peft):
    """LoHa (Low-Rank and Hadamard Product Adaptation) for Linear layers"""

    def __init__(self, loha_rank=4) -> None:
        """Initialize LoHa for Linear layers

        Args:
            rank (int, optional): Rank of the LoHa matrix. Defaults to 4.
        """
        self.loha_rank = loha_rank
        self.layers: list[LoHaLinear] = []
    
    def __call__(self, in_features: int, out_features: int) -> nn.Linear:
        """Create a Linear layer with LoHa and add it to the list of layers

        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features

        Returns:
            nn.Linear: Linear layer with LoHa
        """
        layer = LoHaLinear(in_features, out_features, loha_rank=self.loha_rank)
        self.layers.append(layer)
        return layer

    def state_dict(self) -> dict[str, nn.Parameter]:
        """Get the state dictionary of the LoHa layers for saving
        You can use torch.save(peft.state_dict(), "finetune.pth")

        Returns:
            dict[str, nn.Parameter]: State dictionary of the LoHa layers
        """
        return {
            f"layers.{i}.peft_loha_B1": layer.peft_loha_B1
            for i, layer in enumerate(self.layers)
        } | {
            f"layers.{i}.peft_loha_A1": layer.peft_loha_A1
            for i, layer in enumerate(self.layers)
        } | {
            f"layers.{i}.peft_loha_B2": layer.peft_loha_B2
            for i, layer in enumerate(self.layers)
        } | {
            f"layers.{i}.peft_loha_A2": layer.peft_loha_A2
            for i, layer in enumerate(self.layers)
        } | {
            f"layers.{i}.peft_loha_gamma_value": layer.peft_loha_gamma_value
            for i, layer in enumerate(self.layers)
        }
    
    def load_state_dict(self, state_dict: dict[str, nn.Parameter]) -> None:
        """Load the state dictionary of the LoHa layers

        Args:
            state_dict (dict[str, nn.Parameter]): State dictionary of the LoHa layers
        """
        for i, layer in enumerate(self.layers):
            layer.peft_loha_B1 = state_dict[f"layers.{i}.peft_loha_B1"]
            layer.peft_loha_A1 = state_dict[f"layers.{i}.peft_loha_A1"]
            layer.peft_loha_B2 = state_dict[f"layers.{i}.peft_loha_B2"]
            layer.peft_loha_A2 = state_dict[f"layers.{i}.peft_loha_A2"]
            layer.peft_loha_gamma_value = state_dict[f"layers.{i}.peft_loha_gamma_value"]
