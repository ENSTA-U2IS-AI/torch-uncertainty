from typing import Literal

import torch
from einops import rearrange

from .std import _WideResNet

__all__ = [
    "mimo_wideresnet28x10",
]


class _MIMOWideResNet(_WideResNet):
    def __init__(
        self,
        depth: int,
        widen_factor: int,
        in_channels: int,
        num_classes: int,
        num_estimators: int,
        conv_bias: bool,
        dropout_rate: float,
        groups: int = 1,
        style: Literal["imagenet", "cifar"] = "imagenet",
    ) -> None:
        super().__init__(
            depth,
            widen_factor=widen_factor,
            in_channels=in_channels * num_estimators,
            num_classes=num_classes * num_estimators,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            groups=groups,
            style=style,
        )

        self.num_estimators = num_estimators

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            x = x.repeat(self.num_estimators, 1, 1, 1)
        out = rearrange(x, "(m b) c h w -> b (m c) h w", m=self.num_estimators)
        out = super().forward(out)
        return rearrange(out, "b (m d) -> (m b) d", m=self.num_estimators)


def mimo_wideresnet28x10(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    groups: int = 1,
    conv_bias: bool = True,
    dropout_rate: float = 0.3,
    style: Literal["imagenet", "cifar"] = "imagenet",
) -> _MIMOWideResNet:
    return _MIMOWideResNet(
        depth=28,
        widen_factor=10,
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
    )
