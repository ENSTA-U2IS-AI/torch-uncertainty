from collections.abc import Callable

import torch.nn.functional as F
from torch import Tensor, nn

from torch_uncertainty.layers.bayesian import BayesLinear
from torch_uncertainty.layers.packed import PackedLinear
from torch_uncertainty.models import StochasticModel

__all__ = ["bayesian_mlp", "mlp", "packed_mlp"]


class _MLP(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_outputs: int,
        hidden_dims: list[int],
        layer: type[nn.Module],
        activation: Callable,
        layer_args: dict,
        final_layer: type[nn.Module],
        final_layer_args: dict,
        dropout_rate: float,
    ) -> None:
        """Multi-layer perceptron class.

        Args:
            in_features (int): Number of input features.
            num_outputs (int): Number of output features.
            hidden_dims (list[int]): Number of features for each hidden layer.
            layer (nn.Module): Layer class.
            activation (Callable): Activation function.
            layer_args (Dict): Arguments for the layer class.
            final_layer (nn.Module): Final layer class for distribution regression.
            final_layer_args (Dict): Arguments for the final layer class.
            dropout_rate (float): Dropout probability.
        """
        super().__init__()
        self.activation = activation
        self.dropout_rate = dropout_rate
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
        self.final_layer = final_layer(**final_layer_args)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = F.dropout(layer(x), p=self.dropout_rate, training=self.training)
            x = self.activation(x)
        return self.final_layer(self.layers[-1](x))


def _mlp(
    stochastic: bool,
    in_features: int,
    num_outputs: int,
    hidden_dims: list[int],
    num_samples: int = 16,
    layer_args: dict | None = None,
    layer: type[nn.Module] = nn.Linear,
    activation: Callable = F.relu,
    final_layer: type[nn.Module] = nn.Identity,
    final_layer_args: dict | None = None,
    dropout_rate: float = 0.0,
) -> _MLP | StochasticModel:
    if layer_args is None:
        layer_args = {}
    if final_layer_args is None:
        final_layer_args = {}
    model = _MLP(
        in_features=in_features,
        num_outputs=num_outputs,
        hidden_dims=hidden_dims,
        layer_args=layer_args,
        layer=layer,
        activation=activation,
        final_layer=final_layer,
        final_layer_args=final_layer_args,
        dropout_rate=dropout_rate,
    )
    if stochastic:
        return StochasticModel(model, num_samples)
    return model


def mlp(
    in_features: int,
    num_outputs: int,
    hidden_dims: list[int],
    layer: type[nn.Module] = nn.Linear,
    activation: Callable = F.relu,
    final_layer: type[nn.Module] = nn.Identity,
    final_layer_args: dict | None = None,
    dropout_rate: float = 0.0,
) -> _MLP:
    """Multi-layer perceptron.

    Args:
        in_features (int): Number of input features.
        num_outputs (int): Number of output features.
        hidden_dims (list[int]): Number of features in each hidden layer.
        layer (nn.Module, optional): Layer type. Defaults to nn.Linear.
        activation (Callable, optional): Activation function. Defaults to
            F.relu.
        final_layer (nn.Module, optional): Final layer class for distribution
            regression. Defaults to nn.Identity.
        final_layer_args (Dict, optional): Arguments for the final layer class.
        dropout_rate (float, optional): Dropout probability. Defaults to 0.0.

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
        final_layer=final_layer,
        final_layer_args=final_layer_args,
        dropout_rate=dropout_rate,
    )


def packed_mlp(
    in_features: int,
    num_outputs: int,
    hidden_dims: list[int],
    num_estimators: int = 4,
    alpha: float = 2,
    gamma: float = 1,
    activation: Callable = F.relu,
    final_layer: type[nn.Module] = nn.Identity,
    final_layer_args: dict | None = None,
    dropout_rate: float = 0.0,
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
        final_layer=final_layer,
        final_layer_args=final_layer_args,
        dropout_rate=dropout_rate,
    )


def bayesian_mlp(
    in_features: int,
    num_outputs: int,
    hidden_dims: list[int],
    num_samples: int = 16,
    activation: Callable = F.relu,
    final_layer: type[nn.Module] = nn.Identity,
    final_layer_args: dict | None = None,
    dropout_rate: float = 0.0,
) -> StochasticModel:
    return _mlp(
        stochastic=True,
        num_samples=num_samples,
        in_features=in_features,
        num_outputs=num_outputs,
        hidden_dims=hidden_dims,
        layer=BayesLinear,
        activation=activation,
        final_layer=final_layer,
        final_layer_args=final_layer_args,
        dropout_rate=dropout_rate,
    )
