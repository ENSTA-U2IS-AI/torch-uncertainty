from collections.abc import Callable

import torch.nn.functional as F
from torch import Tensor, nn

from torch_uncertainty.layers.bayesian import BayesLinear
from torch_uncertainty.layers.distributions import get_dist_linear_layer
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
        dropout_rate: float,
        dist_family: str | None,
        dist_args: dict,
    ) -> None:
        """Multi-layer perceptron class.

        Args:
            in_features (int): Number of input features.
            num_outputs (int): Number of output features.
            hidden_dims (list[int]): Number of features for each hidden layer.
            layer (nn.Module): Layer class.
            activation (Callable): Activation function.
            layer_args (Dict): Arguments for the layer class.
            dropout_rate (float): Dropout probability.
            dist_family (str, optional): Distribution family name. ``None`` means point-wise
                prediction. Defaults to ``None``.
            dist_args (Dict, optional): Arguments for the distribution layer class.
        """
        super().__init__()
        self.activation = activation
        self.dropout_rate = dropout_rate
        layers = nn.ModuleList()

        if len(hidden_dims) == 0:
            if layer == PackedLinear:
                layer_args |= {"first": True, "last": True}

            self.final_layer = layer(
                in_features=in_features, out_features=num_outputs, **layer_args
            )
        else:
            if layer == PackedLinear:
                layer_args |= {"first": True, "last": False}

            layers.append(layer(in_features=in_features, out_features=hidden_dims[0], **layer_args))

            if layer == PackedLinear:
                layer_args |= {"first": False}

            for i in range(1, len(hidden_dims)):
                layers.append(
                    layer(in_features=hidden_dims[i - 1], out_features=hidden_dims[i], **layer_args)
                )

            if layer == PackedLinear:
                layer_args |= {"last": True}

            if dist_family is not None:
                dist_layer_class = get_dist_linear_layer(dist_family)
                self.final_layer = dist_layer_class(
                    base_layer=layer,
                    event_dim=num_outputs,
                    in_features=hidden_dims[-1],
                    **layer_args,
                    **dist_args,
                )
            else:
                self.final_layer = layer(
                    in_features=hidden_dims[-1], out_features=num_outputs, **layer_args
                )

        self.layers = layers

    def forward(self, x: Tensor) -> Tensor | dict[str, Tensor]:
        for layer in self.layers:
            x = F.dropout(layer(x), p=self.dropout_rate, training=self.training)
            x = self.activation(x)
        return self.final_layer(x)


def _mlp(
    stochastic: bool,
    in_features: int,
    num_outputs: int,
    hidden_dims: list[int],
    num_samples: int = 16,
    layer: type[nn.Module] = nn.Linear,
    layer_args: dict | None = None,
    activation: Callable = F.relu,
    dropout_rate: float = 0.0,
    dist_family: str | None = None,
    dist_args: dict | None = None,
) -> _MLP | StochasticModel:
    model = _MLP(
        in_features=in_features,
        num_outputs=num_outputs,
        hidden_dims=hidden_dims,
        layer_args=layer_args or {},
        layer=layer,
        activation=activation,
        dropout_rate=dropout_rate,
        dist_family=dist_family,
        dist_args=dist_args or {},
    )
    if stochastic:
        return StochasticModel(model, num_samples)
    return model


def mlp(
    in_features: int,
    num_outputs: int,
    hidden_dims: list[int],
    activation: Callable = F.relu,
    dropout_rate: float = 0.0,
    dist_family: str | None = None,
    dist_args: dict | None = None,
) -> _MLP:
    """Multi-layer perceptron.

    Args:
        in_features (int): Number of input features.
        num_outputs (int): Number of output features.
        hidden_dims (list[int]): Number of features in each hidden layer.
        activation (Callable, optional): Activation function. Defaults to
            F.relu.
        dropout_rate (float, optional): Dropout probability. Defaults to 0.0.
        dist_family (str, optional): Distribution family. Defaults to None.
        dist_args (Dict, optional): Arguments for the distribution layer class. Defaults
            to None.

    Returns:
        _MLP: A Multi-Layer-Perceptron model.
    """
    return _mlp(
        stochastic=False,
        in_features=in_features,
        num_outputs=num_outputs,
        hidden_dims=hidden_dims,
        activation=activation,
        dropout_rate=dropout_rate,
        dist_family=dist_family,
        dist_args=dist_args,
    )


def packed_mlp(
    in_features: int,
    num_outputs: int,
    hidden_dims: list[int],
    num_estimators: int = 4,
    alpha: float = 2,
    gamma: float = 1,
    activation: Callable = F.relu,
    dropout_rate: float = 0.0,
    dist_family: str | None = None,
    dist_args: dict | None = None,
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
        dropout_rate=dropout_rate,
        dist_family=dist_family,
        dist_args=dist_args,
    )


def bayesian_mlp(
    in_features: int,
    num_outputs: int,
    hidden_dims: list[int],
    num_samples: int = 16,
    activation: Callable = F.relu,
    dropout_rate: float = 0.0,
    dist_family: str | None = None,
    dist_args: dict | None = None,
) -> StochasticModel:
    return _mlp(
        stochastic=True,
        num_samples=num_samples,
        in_features=in_features,
        num_outputs=num_outputs,
        hidden_dims=hidden_dims,
        layer=BayesLinear,
        activation=activation,
        dropout_rate=dropout_rate,
        dist_family=dist_family,
        dist_args=dist_args,
    )
