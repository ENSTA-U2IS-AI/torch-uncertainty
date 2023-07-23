# fmt: off
from typing import Callable, Dict, Optional, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.bayesian_layers import BayesConv2d, BayesLinear
from ..layers.packed_layers import PackedConv2d, PackedLinear
from .utils import StochasticModel

# fmt: on
__all__ = ["lenet", "packed_lenet", "bayesian_lenet"]


class _LeNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        linear_layer: Type[nn.Module],
        conv2d_layer: Type[nn.Module],
        layer_args: Dict,
        activation: Callable,
        norm: Type[nn.Module],
        groups: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.activation = activation
        self.norm = norm()  # TODO: Fix when not Identity
        self.dropout = dropout

        self.conv1 = conv2d_layer(
            in_channels, 6, (5, 5), groups=groups, **layer_args
        )
        self.conv2 = conv2d_layer(6, 16, (5, 5), groups=groups, **layer_args)
        self.pooling = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = linear_layer(256, 120, **layer_args)
        self.fc2 = linear_layer(120, 84, **layer_args)
        self.fc3 = linear_layer(84, num_classes, **layer_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.dropout(
            self.activation(self.norm(self.conv1(x))),
            p=self.dropout,
            training=self.training,
        )
        out = F.max_pool2d(out, 2)
        out = F.dropout(
            self.activation(self.norm(self.conv2(out))),
            p=self.dropout,
            training=self.training,
        )
        out = F.max_pool2d(out, 2)
        out = self.pooling(out)
        out = torch.flatten(out, 1)
        out = F.dropout(
            self.activation(self.norm(self.fc1(out))),
            p=self.dropout,
            training=self.training,
        )
        out = F.dropout(
            self.activation(self.norm(self.fc2(out))),
            p=self.dropout,
            training=self.training,
        )
        out = self.fc3(out)
        return out


@StochasticModel
class _StochasticLeNet(_LeNet):
    pass


def _lenet(
    stochastic: bool,
    in_channels: int,
    num_classes: int,
    linear_layer: Type[nn.Module] = nn.Linear,
    conv2d_layer: Type[nn.Module] = nn.Conv2d,
    layer_args: Dict = {},
    activation: Callable = nn.ReLU,
    norm: Type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout: float = 0.0,
) -> Union[_LeNet, _StochasticLeNet]:
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
        dropout=dropout,
    )


def lenet(
    in_channels: int,
    num_classes: int,
    activation: Callable = F.relu,
    norm: Type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout: float = 0.0,
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
        dropout=dropout,
    )


def packed_lenet(
    in_channels: int,
    num_classes: int,
    num_estimators: int = 4,
    alpha: float = 2,
    gamma: float = 1,
    activation: Callable = F.relu,
    norm: Type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout: float = 0.0,
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
        dropout=dropout,
    )


def bayesian_lenet(
    in_channels: int,
    num_classes: int,
    prior_mu: Optional[float] = None,
    prior_sigma_1: Optional[float] = None,
    prior_sigma_2: Optional[float] = None,
    prior_pi: Optional[float] = None,
    mu_init: Optional[float] = None,
    sigma_init: Optional[float] = None,
    activation: Callable = F.relu,
    norm: Type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout: float = 0.0,
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
        dropout=dropout,
    )
