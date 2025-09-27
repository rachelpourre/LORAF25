from pathlib import Path

import torch

from .bignet import BIGNET_DIM, LayerNorm  # noqa: F401
from .half_precision import HalfLinear
import torch.nn as nn
import torch.nn.functional as functional


class LoRALinear(HalfLinear):

    def __init__(
        self,
        in_features: int,
        out_features: int,
        lora_dim: int,
        bias: bool = True,
    ) -> None:
        """
        Implement the LoRALinear layer as described in the homework

        Hint: You can use the HalfLinear class as a parent class (it makes load_state_dict easier, names match)
        Hint: Remember to initialize the weights of the lora layers
        Hint: Make sure the linear layers are not trainable, but the LoRA layers are
        """
        super().__init__(in_features, out_features, bias=bias)

        if hasattr(self, "weight") and isinstance(self.weight, torch.nn.Parameter):
            self.weight.requires_grad_(False)
        if hasattr(self, "bias") and isinstance(self.bias, torch.nn.Parameter) and self.bias is not None:
            self.bias.requires_grad_(False)

        self.lora_A = nn.Linear(in_features, lora_dim, bias=False) 
        self.lora_B = nn.Linear(lora_dim, out_features, bias=False) 

        nn.init.kaiming_uniform_(self.lora_A.weight, a=(5.0 ** 0.5))
        nn.init.zeros_(self.lora_B.weight)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = super().forward(x)

        lora_input = x if x.dtype == torch.float32 else x.float()
        lora_out = self.lora_B(self.lora_A(lora_input))

        if lora_out.dtype != base_out.dtype:
            lora_out = lora_out.to(base_out.dtype)

        return base_out + lora_out

class LoraBigNet(torch.nn.Module):
    class Block(torch.nn.Module):
        def __init__(self, channels: int, lora_dim: int):
            super().__init__()
            self.model = torch.nn.Sequential(
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
                torch.nn.ReLU(),
                LoRALinear(channels, channels, lora_dim),
            )


        def forward(self, x: torch.Tensor):
            return self.model(x) + x

    def __init__(self, lora_dim: int = 32):
        super().__init__()
        self.model = torch.nn.Sequential(
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
            LayerNorm(BIGNET_DIM),
            self.Block(BIGNET_DIM, lora_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load(path: Path | None) -> LoraBigNet:
    # Since we have additional layers, we need to set strict=False in load_state_dict
    net = LoraBigNet()
    if path is not None:
        net.load_state_dict(torch.load(path, weights_only=True), strict=False)
    return net
