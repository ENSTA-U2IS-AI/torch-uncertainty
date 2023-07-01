from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..layers.bayesian_layers import BayesConv2d, BayesLinear
from ..layers.packed_layers import PackedConv2d, PackedLinear
from .utils import Stochastic


class _LeNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        linear_layer: nn.Module,
        conv2d_layer: nn.Module,
        # activation: Callable,
        layer_args: Dict,
        dropout: float,
    ) -> None:
        super().__init__()
        self.conv1 = conv2d_layer(in_channels, 6, (5, 5))
        self.conv2 = conv2d_layer(6, 16, (5, 5))
        self.pooling = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = linear_layer(256, 120)
        self.fc2 = linear_layer(120, 84)
        self.fc3 = linear_layer(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = self.pooling(out)
        out = torch.flatten(out, 1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


@Stochastic
class _StochasticLeNet(_LeNet):
    pass


def _lenet(
    stochastic: bool,
    in_channels: int,
    num_classes: int,
    linear_layer: nn.Module = nn.Linear,
    conv2d_layer: nn.Module = nn.Conv2d,
    norm: Optional[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout: float = 0.0,
    **model_kwargs: Any,
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
        # norm,
        # groups,
        layer_args=model_kwargs,
        dropout=dropout,
    )


def lenet(
    in_channels: int,
    num_classes: int,
    norm: Optional[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout: float = 0.0,
    **model_kwargs: Any,
) -> _LeNet:
    return _lenet(
        False,
        in_channels=in_channels,
        num_classes=num_classes,
        linear_layer=nn.Linear,
        conv2d_layer=nn.Conv2d,
        # norm,
        # groups,
        layer_args={},
        dropout=dropout,
        **model_kwargs,
    )


def packed_lenet(
    in_channels: int,
    num_classes: int,
    norm: Optional[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout: float = 0.0,
    **model_kwargs: Any,
) -> _LeNet:
    return _lenet(
        stochastic=False,
        in_channels=in_channels,
        num_classes=num_classes,
        linear_layer=PackedLinear,
        conv2d_layer=PackedConv2d,
        # norm,
        # groups,
        layer_args={},
        dropout=dropout,
        **model_kwargs,
    )


def bayesian_lenet(
    in_channels: int,
    num_classes: int,
    norm: Optional[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout: float = 0.0,
    **model_kwargs: Any,
) -> _LeNet:
    return _lenet(
        stochastic=True,
        in_channels=in_channels,
        num_classes=num_classes,
        linear_layer=BayesLinear,
        conv2d_layer=BayesConv2d,
        # norm,
        # groups,
        layer_args={},
        dropout=dropout,
        **model_kwargs,
    )
