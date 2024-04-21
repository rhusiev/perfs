import torch
from torch import nn
from torch.nn.init import kaiming_uniform_

from pefts.peft import Peft


class LoKrLinear(nn.Linear):
    """A single Linear layer with LoKr.

    (Low-rank adaptation with Kronecker product).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        lokr_rank: int = 4,
        kaiming: bool = True,
        factor: int = 8,
    ) -> None:
        """Initialize a Linear layer with LoKr.

        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features
            bias (bool, optional): Whether to include bias. Defaults to True.
            lokr_rank (int, optional): Rank of the B*A LoKr matrix. Defaults to 4.
            kaiming (bool, optional): Whether to initialize with Kaiming uniform.
                If False, initializes with standard normal. Defaults to True.
            factor (int, optional): Factor for the formula
                u_p = max(u <= min(factor, sqrt(p)) | p mod u = 0)
                where p - in_features
                With lower factor, gives more quality
                Defaults to 8.
        """
        super().__init__(in_features, out_features, bias=bias)
        self.factor = factor
        self.lokr_rank = lokr_rank

        u_p_max_allowed = min(self.factor, int(in_features**0.5))
        self.u_p = 1
        for u in range(u_p_max_allowed, 0, -1):
            if in_features % u == 0:
                self.u_p = u
                break
        self.v_p = in_features // self.u_p

        self.u_q = 1
        for u in range(u_p_max_allowed, 0, -1):
            if out_features % u == 0:
                self.u_q = u
                break
        self.v_q = out_features // self.u_q

        self.peft_lokr_C = nn.Parameter(torch.zeros(self.u_p, self.u_q))
        self.peft_lokr_A = nn.Parameter(torch.zeros(self.lokr_rank, self.v_q))
        self.peft_lokr_B = nn.Parameter(torch.zeros(self.v_p, self.lokr_rank))

        if kaiming:
            kaiming_uniform_(self.peft_lokr_C, a=5**0.5)
        else:
            nn.init.normal_(self.peft_lokr_C, std=1.0 / self.u_p)

        nn.init.zeros_(self.peft_lokr_A)
        nn.init.zeros_(self.peft_lokr_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the Linear layer with LoKr.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features)

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, out_features)
        """
        # Apply LoKr
        BA = self.peft_lokr_B @ self.peft_lokr_A
        weight = self.weight + torch.kron(self.peft_lokr_C, BA)
        # Apply Linear layer
        return nn.functional.linear(x, weight, self.bias)

    def to(self, *args, **kwargs) -> "LoKrLinear":
        """Move the LoKrLinear layer to a device.

        Args:
            *args: Arguments for the to method
            **kwargs: Keyword arguments for the to method

        Returns:
            LoKrLinear: Moved LoKrLinear layer
        """
        self.peft_lokr_A = self.peft_lokr_A.to(*args, **kwargs)
        self.peft_lokr_B = self.peft_lokr_B.to(*args, **kwargs)
        self.peft_lokr_C = self.peft_lokr_C.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class LoKrPeft(Peft):
    """LoKr (Low-rank adaptation with Kronecker product) for Linear layers."""

    def __init__(self, lokr_rank: int = 4, factor: int = 8) -> None:
        """Initialize LoKr for Linear layers.

        Args:
            lokr_rank (int, optional): Rank of the B*A LoKr matrix. Defaults to 4.
            factor (int, optional): Factor for the formula
                u_p = max(u <= min(factor, sqrt(p)) | p mod u = 0)
                where p - in_features
                With lower factor, gives more quality
                Defaults to 8.
        """
        self.lokr_rank = lokr_rank
        self.factor = factor
        self.layers: list[LoKrLinear] = []

    def __call__(self, in_features: int, out_features: int) -> nn.Linear:
        """Create a Linear layer with LoKr and add it to the list of layers.

        Args:
            in_features (int): Number of input features
            out_features (int): Number of output features

        Returns:
            nn.Linear: Linear layer with LoKr
        """
        layer = LoKrLinear(
            in_features, out_features, lokr_rank=self.lokr_rank, factor=self.factor
        )
        self.layers.append(layer)
        return layer

    def state_dict(self) -> dict[str, nn.Parameter]:
        """Get the state dictionary of the LoKr layers for saving You can use
        torch.save(peft.state_dict(), "finetune.pth")

        Returns:
            dict[str, nn.Parameter]: State dictionary of the LoKr layers
        """
        return {
            f"layers.{i}.peft_lokr_A": layer.peft_lokr_A
            for i, layer in enumerate(self.layers)
        } | {
            f"layers.{i}.peft_lokr_B": layer.peft_lokr_B
            for i, layer in enumerate(self.layers)
        } | {
            f"layers.{i}.peft_lokr_C": layer.peft_lokr_C
            for i, layer in enumerate(self.layers)
        }

    def load_state_dict(self, state_dict: dict[str, nn.Parameter]) -> None:
        """Load the state dictionary of the LoKr layers.

        Args:
            state_dict (dict[str, nn.Parameter]): State dictionary of the LoKr layers
        """
        for i, layer in enumerate(self.layers):
            layer.peft_lokr_A = state_dict[f"layers.{i}.peft_lokr_A"]
            layer.peft_lokr_B = state_dict[f"layers.{i}.peft_lokr_B"]
            layer.peft_lokr_C = state_dict[f"layers.{i}.peft_lokr_C"]
