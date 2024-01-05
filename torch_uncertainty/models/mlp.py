from collections.abc import Callable

import torch.nn.functional as F
from torch import Tensor, nn

from torch_uncertainty.layers.bayesian import BayesLinear
from torch_uncertainty.layers.packed import PackedLinear
from torch_uncertainty.models.utils import stochastic_model

__all__ = ["mlp", "packed_mlp", "bayesian_mlp"]


class _MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_outputs: int,
        hidden_dims: list[int],
        layer: type[nn.Module],
        activation: Callable,
        layer_args: dict,
        dropout: float,
    ) -> None:
        """Multi-layer perceptron class.

        Args:
            in_features (int): Number of input features.
            num_outputs (int): Number of output features.
            hidden_dims (List[int]): Number of features for each hidden layer.
            layer (nn.Module): Layer class.
            activation (Callable): Activation function.
            layer_args (Dict): Arguments for the layer class.
            dropout (float): Dropout probability.
        """
        super().__init__()
        self.activation = activation
        self.dropout = dropout

        layers = nn.ModuleList()

        if len(hidden_dims) == 0:
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
                    layer(in_features, hidden_dims[0], first=True, **layer_args)
                )
            else:
                layers.append(layer(in_features, hidden_dims[0], **layer_args))

            for i in range(1, len(hidden_dims)):
                layers.append(
                    layer(hidden_dims[i - 1], hidden_dims[i], **layer_args)
                )

            if layer == PackedLinear:
                layers.append(
                    layer(hidden_dims[-1], num_outputs, last=True, **layer_args)
                )
            else:
                layers.append(layer(hidden_dims[-1], num_outputs, **layer_args))

        self.layers = layers

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = F.dropout(layer(x), p=self.dropout, training=self.training)
            x = self.activation(x)
        return self.layers[-1](x)


@stochastic_model
class _StochasticMLP(_MLP):
    pass


def _mlp(
    stochastic: bool,
    in_features: int,
    num_outputs: int,
    hidden_dims: list[int],
    layer_args: dict | None = None,
    layer: type[nn.Module] = nn.Linear,
    activation: Callable = F.relu,
    dropout: float = 0.0,
) -> _MLP | _StochasticMLP:
    if layer_args is None:
        layer_args = {}
    model = _MLP if not stochastic else _StochasticMLP
    return model(
        in_features=in_features,
        num_outputs=num_outputs,
        hidden_dims=hidden_dims,
        layer_args=layer_args,
        layer=layer,
        activation=activation,
        dropout=dropout,
    )


def mlp(
    in_features: int,
    num_outputs: int,
    hidden_dims: list[int],
    layer: type[nn.Module] = nn.Linear,
    activation: Callable = F.relu,
    dropout: float = 0.0,
) -> _MLP:
    """Multi-layer perceptron.

    Args:
        in_features (int): Number of input features.
        num_outputs (int): Number of output features.
        hidden_dims (List[int]): Number of features in each hidden layer.
        layer (nn.Module, optional): Layer type. Defaults to nn.Linear.
        activation (Callable, optional): Activation function. Defaults to
            F.relu.
        dropout (float, optional): Dropout probability. Defaults to 0.0.

    Returns:
        _MLP: A Multi-Layer-Perceptron model.
    """
    return _mlp(
        stochastic=False,
        in_features=in_features,
        num_outputs=num_outputs,
        hidden_dims=hidden_dims,
        layer=layer,
        activation=activation,
        dropout=dropout,
    )


def packed_mlp(
    in_features: int,
    num_outputs: int,
    hidden_dims: list[int],
    num_estimators: int = 4,
    alpha: float = 2,
    gamma: float = 1,
    activation: Callable = F.relu,
    dropout: float = 0.0,
) -> _MLP:
    layer_args = {
        "num_estimators": num_estimators,
        "alpha": alpha,
        "gamma": gamma,
    }
    return _mlp(
        stochastic=False,
        in_features=in_features,
        num_outputs=num_outputs,
        hidden_dims=hidden_dims,
        layer=PackedLinear,
        activation=activation,
        layer_args=layer_args,
        dropout=dropout,
    )


def bayesian_mlp(
    in_features: int,
    num_outputs: int,
    hidden_dims: list[int],
    activation: Callable = F.relu,
    dropout: float = 0.0,
) -> _StochasticMLP:
    return _mlp(
        stochastic=True,
        in_features=in_features,
        num_outputs=num_outputs,
        hidden_dims=hidden_dims,
        layer=BayesLinear,
        activation=activation,
        dropout=dropout,
    )
