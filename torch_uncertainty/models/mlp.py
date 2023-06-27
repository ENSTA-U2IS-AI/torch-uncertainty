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
    """Multi-layer perceptron class.

    Args:
        in_features (int): Number of input features.
        num_outputs (int): Number of output features.
        hidden_dim (List[int]): Number of features for each hidden layer.
        layer (nn.Module): Layer class.
        activation (Callable): Activation function.
        layer_args (Dict): Arguments for the layer class.
    """

    def __init__(
        self,
        in_features: int,
        num_outputs: int,
        hidden_dim: List[int],
        layer: nn.Module,
        activation: Callable,
        layer_args: Dict,
    ) -> None:
        super().__init__()
        self.activation = activation

        layers = nn.ModuleList()

        if len(hidden_dim) == 0:
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
                    layer(in_features, hidden_dim[0], first=True, **layer_args)
                )
            else:
                layers.append(layer(in_features, hidden_dim[0], **layer_args))

            for i in range(1, len(hidden_dim)):
                layers.append(
                    layer(hidden_dim[i - 1], hidden_dim[i], **layer_args)
                )

            if layer == PackedLinear:
                layers.append(
                    layer(hidden_dim[-1], num_outputs, last=True, **layer_args)
                )
            else:
                layers.append(layer(hidden_dim[-1], num_outputs, **layer_args))
        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        out = self.layers[-1](x)
        return out


def mlp(
    in_features: int,
    num_outputs: int,
    hidden_dim: List[int],
    layer: nn.Module = nn.Linear,
    activation: Callable = F.relu,
    layer_args: dict = {},
) -> _MLP:
    """Multi-layer perceptron.

    Args:
        in_features (int): Number of input features.
        num_outputs (int): Number of output features.
        hidden_dim (List[int]): Number of features in each hidden layer.
        layer (nn.Module, optional): Layer type. Defaults to nn.Linear.
        activation (Callable, optional): Activation function. Defaults to
            F.relu.
        layer_args (dict, optional): Arguments for the layer. Defaults to {}.

    Returns:
        _MLP: A Multi-Layer-Perceptron model.
    """
    return _MLP(
        in_features=in_features,
        num_outputs=num_outputs,
        hidden_dim=hidden_dim,
        layer=layer,
        activation=activation,
        layer_args=layer_args,
    )
