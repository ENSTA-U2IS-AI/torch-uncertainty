from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..layers.bayesian import BayesConv2d, BayesLinear
from ..layers.packed import PackedConv2d, PackedLinear
from .utils import StochasticModel, toggle_dropout

__all__ = ["lenet", "packed_lenet", "bayesian_lenet"]


class _LeNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        linear_layer: type[nn.Module],
        conv2d_layer: type[nn.Module],
        layer_args: dict,
        activation: Callable,
        norm: type[nn.Module],
        groups: int,
        dropout_rate: float,
        num_estimators: int,
        last_layer_dropout: bool,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.norm = norm()  # TODO: Fix when not Identity
        self.dropout_rate = dropout_rate
        self.num_estimators = num_estimators
        self.last_layer_dropout = last_layer_dropout

        self.conv1 = conv2d_layer(
            in_channels, 6, (5, 5), groups=groups, **layer_args
        )
        self.conv2 = conv2d_layer(6, 16, (5, 5), groups=groups, **layer_args)
        self.pooling = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = linear_layer(256, 120, **layer_args)
        self.fc2 = linear_layer(120, 84, **layer_args)
        self.fc3 = linear_layer(84, num_classes, **layer_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.handle_dropout(x)
        out = F.dropout(
            self.activation(self.norm(self.conv1(x))),
            p=self.dropout_rate,
        )
        out = F.max_pool2d(out, 2)
        out = F.dropout(
            self.activation(self.norm(self.conv2(out))),
            p=self.dropout_rate,
        )
        out = F.max_pool2d(out, 2)
        out = self.pooling(out)
        out = torch.flatten(out, 1)
        out = F.dropout(
            self.activation(self.norm(self.fc1(out))),
            p=self.dropout_rate,
        )
        out = F.dropout(
            self.activation(self.norm(self.fc2(out))),
            p=self.dropout_rate,
        )
        return self.fc3(out)

    def handle_dropout(self, x: Tensor) -> Tensor:
        if self.num_estimators is not None:
            if not self.training:
                if self.last_layer_dropout is not None:
                    toggle_dropout(self, self.last_layer_dropout)
                x = x.repeat(self.num_estimators, 1, 1, 1)
        return x


@StochasticModel
class _StochasticLeNet(_LeNet):
    pass


def _lenet(
    stochastic: bool,
    in_channels: int,
    num_classes: int,
    linear_layer: type[nn.Module] = nn.Linear,
    conv2d_layer: type[nn.Module] = nn.Conv2d,
    layer_args: dict = {},
    activation: Callable = nn.ReLU,
    norm: type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout_rate: float = 0.0,
    num_estimators: int | None = None,
    last_layer_dropout: bool = False,
) -> _LeNet | _StochasticLeNet:
    if not stochastic:
        model = _LeNet
    else:
        model = _StochasticLeNet
    return model(
        in_channels=in_channels,
        num_classes=num_classes,
        linear_layer=linear_layer,
        conv2d_layer=conv2d_layer,
        activation=activation,
        norm=norm,
        groups=groups,
        layer_args=layer_args,
        dropout_rate=dropout_rate,
        num_estimators=num_estimators,
        last_layer_dropout=last_layer_dropout,
    )


def lenet(
    in_channels: int,
    num_classes: int,
    activation: Callable = F.relu,
    norm: type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout_rate: float = 0.0,
    num_estimators: int | None = None,
    last_layer_dropout: bool = False,
) -> _LeNet:
    return _lenet(
        False,
        in_channels=in_channels,
        num_classes=num_classes,
        linear_layer=nn.Linear,
        conv2d_layer=nn.Conv2d,
        layer_args={},
        activation=activation,
        norm=norm,
        groups=groups,
        dropout_rate=dropout_rate,
        num_estimators=num_estimators,
        last_layer_dropout=last_layer_dropout,
    )


def packed_lenet(
    in_channels: int,
    num_classes: int,
    num_estimators: int = 4,
    alpha: float = 2,
    gamma: float = 1,
    activation: Callable = F.relu,
    norm: type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout_rate: float = 0.0,
) -> _LeNet:
    return _lenet(
        stochastic=False,
        in_channels=in_channels,
        num_classes=num_classes,
        linear_layer=PackedLinear,
        conv2d_layer=PackedConv2d,
        norm=norm,
        layer_args={
            "num_estimators": num_estimators,
            "alpha": alpha,
            "gamma": gamma,
        },
        activation=activation,
        groups=groups,
        dropout_rate=dropout_rate,
    )


def bayesian_lenet(
    in_channels: int,
    num_classes: int,
    prior_mu: float | None = None,
    prior_sigma_1: float | None = None,
    prior_sigma_2: float | None = None,
    prior_pi: float | None = None,
    mu_init: float | None = None,
    sigma_init: float | None = None,
    activation: Callable = F.relu,
    norm: type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout_rate: float = 0.0,
) -> _LeNet:
    layers_args = {}
    if prior_mu is not None:
        layers_args["prior_mu"] = prior_mu
    if prior_sigma_1 is not None:
        layers_args["prior_sigma_1"] = prior_sigma_1
    if prior_sigma_2 is not None:
        layers_args["prior_sigma_2"] = prior_sigma_2
    if prior_pi is not None:
        layers_args["prior_pi"] = prior_pi
    if mu_init is not None:
        layers_args["mu_init"] = mu_init
    if sigma_init is not None:
        layers_args["sigma_init"] = sigma_init

    return _lenet(
        stochastic=True,
        in_channels=in_channels,
        num_classes=num_classes,
        linear_layer=BayesLinear,
        conv2d_layer=BayesConv2d,
        norm=norm,
        layer_args=layers_args,
        activation=activation,
        groups=groups,
        dropout_rate=dropout_rate,
    )
