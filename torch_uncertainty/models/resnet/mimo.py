# fmt: off
from typing import List, Type, Union

import torch
from einops import rearrange

from .std import BasicBlock, Bottleneck, _ResNet

# fmt: on
__all__ = [
    "mimo_resnet18",
    "mimo_resnet34",
    "mimo_resnet50",
    "mimo_resnet101",
    "mimo_resnet152",
]


class _MIMOResNet(_ResNet):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        num_blocks: List[int],
        in_channels: int,
        num_classes: int,
        num_estimators: int,
        dropout_rate: float = 0.0,
        groups: int = 1,
        style: str = "imagenet",
    ) -> None:
        super().__init__(
            block=block,
            num_blocks=num_blocks,
            in_channels=in_channels * num_estimators,
            num_classes=num_classes * num_estimators,
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
        out = rearrange(out, "b (m d) -> (m b) d", m=self.num_estimators)
        return out


def mimo_resnet18(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    dropout_rate: float = 0.0,
    groups: int = 1,
    style: str = "imagenet",
) -> _MIMOResNet:
    return _MIMOResNet(
        block=BasicBlock,
        num_blocks=[2, 2, 2, 2],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
    )


def mimo_resnet34(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    dropout_rate: float = 0.0,
    groups: int = 1,
    style: str = "imagenet",
) -> _MIMOResNet:
    return _MIMOResNet(
        block=BasicBlock,
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
    )


def mimo_resnet50(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    dropout_rate: float = 0.0,
    groups: int = 1,
    style: str = "imagenet",
) -> _MIMOResNet:
    return _MIMOResNet(
        block=Bottleneck,
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
    )


def mimo_resnet101(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    dropout_rate: float = 0.0,
    groups: int = 1,
    style: str = "imagenet",
) -> _MIMOResNet:
    return _MIMOResNet(
        block=Bottleneck,
        num_blocks=[3, 4, 23, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
    )


def mimo_resnet152(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    dropout_rate: float = 0.0,
    groups: int = 1,
    style: str = "imagenet",
) -> _MIMOResNet:
    return _MIMOResNet(
        block=Bottleneck,
        num_blocks=[3, 8, 36, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
    )
