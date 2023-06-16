# fmt: off
from typing import Callable, Dict, List

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from ..layers.packed_layers import PackedLinear

# fmt: on
__all__ = [
    "mlp",
]


class _MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_outputs: int,
        num_features: List[int],
        layer: nn.Module,
        activation: Callable,
        layer_args: Dict,
    ) -> None:
        super().__init__()
        self.activation = activation

        layers = nn.ModuleList()

        if len(num_features) == 0:
            if layer == PackedLinear:
                layers.append(
                    layer(
                        in_features,
                        num_outputs,
                        first=True,
                        last=True,
                        **layer_args,
                    )
                )
            else:
                layers.append(layer(in_features, num_outputs, **layer_args))
        else:
            if layer == PackedLinear:
                layers.append(
                    layer(
                        in_features, num_features[0], first=True, **layer_args
                    )
                )
            else:
                layers.append(layer(in_features, num_features[0], **layer_args))

            for i in range(1, len(num_features)):
                layers.append(
                    layer(num_features[i - 1], num_features[i], **layer_args)
                )

            if layer == PackedLinear:
                layers.append(
                    layer(
                        num_features[-1], num_outputs, last=True, **layer_args
                    )
                )
            else:
                layers.append(
                    layer(num_features[-1], num_outputs, **layer_args)
                )
        self.layers = layers
        print(self.layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        out = self.layers[-1](x)
        return out


def mlp(
    in_features: int,
    num_outputs: int,
    num_features: List[int],
    layer: nn.Module = nn.Linear,
    activation: Callable = F.relu,
    layer_args: dict = {},
) -> _MLP:
    return _MLP(
        in_features=in_features,
        num_outputs=num_outputs,
        num_features=num_features,
        layer=layer,
        activation=activation,
        layer_args=layer_args,
    )
