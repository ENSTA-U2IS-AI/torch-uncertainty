from typing import Literal

import torch
from einops import rearrange
from torch import nn

from .std import _BasicBlock, _Bottleneck, _ResNet

__all__ = [
    "mimo_resnet18",
    "mimo_resnet20",
    "mimo_resnet34",
    "mimo_resnet50",
    "mimo_resnet101",
    "mimo_resnet152",
]


class _MIMOResNet(_ResNet):
    def __init__(
        self,
        block: type[_BasicBlock | _Bottleneck],
        num_blocks: list[int],
        in_channels: int,
        num_classes: int,
        num_estimators: int,
        conv_bias: bool,
        dropout_rate: float,
        groups: int = 1,
        style: Literal["imagenet", "cifar"] = "imagenet",
        in_planes: int = 64,
        normalization_layer: nn.Module = nn.BatchNorm2d,
    ) -> None:
        super().__init__(
            block=block,
            num_blocks=num_blocks,
            in_channels=in_channels * num_estimators,
            num_classes=num_classes * num_estimators,
            conv_bias=conv_bias,
            dropout_rate=dropout_rate,
            groups=groups,
            style=style,
            in_planes=in_planes,
            normalization_layer=normalization_layer,
        )

        self.num_estimators = num_estimators

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            x = x.repeat(self.num_estimators, 1, 1, 1)

        out = rearrange(x, "(m b) c h w -> b (m c) h w", m=self.num_estimators)
        out = super().forward(out)
        return rearrange(out, "b (m d) -> (m b) d", m=self.num_estimators)


def mimo_resnet18(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    conv_bias: bool = True,
    dropout_rate: float = 0.0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: nn.Module = nn.BatchNorm2d,
) -> _MIMOResNet:
    return _MIMOResNet(
        block=_BasicBlock,
        num_blocks=[2, 2, 2, 2],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=64,
        normalization_layer=normalization_layer,
    )


def mimo_resnet20(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    conv_bias: bool = True,
    dropout_rate: float = 0.0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: nn.Module = nn.BatchNorm2d,
) -> _MIMOResNet:
    return _MIMOResNet(
        block=_BasicBlock,
        num_blocks=[3, 3, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=16,
        normalization_layer=normalization_layer,
    )


def mimo_resnet34(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    conv_bias: bool = True,
    dropout_rate: float = 0.0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: nn.Module = nn.BatchNorm2d,
) -> _MIMOResNet:
    return _MIMOResNet(
        block=_BasicBlock,
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=64,
        normalization_layer=normalization_layer,
    )


def mimo_resnet50(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    conv_bias: bool = True,
    dropout_rate: float = 0.0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: nn.Module = nn.BatchNorm2d,
) -> _MIMOResNet:
    return _MIMOResNet(
        block=_Bottleneck,
        num_blocks=[3, 4, 6, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=64,
        normalization_layer=normalization_layer,
    )


def mimo_resnet101(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    conv_bias: bool = True,
    dropout_rate: float = 0.0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: nn.Module = nn.BatchNorm2d,
) -> _MIMOResNet:
    return _MIMOResNet(
        block=_Bottleneck,
        num_blocks=[3, 4, 23, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=64,
        normalization_layer=normalization_layer,
    )


def mimo_resnet152(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    conv_bias: bool = True,
    dropout_rate: float = 0.0,
    groups: int = 1,
    style: Literal["imagenet", "cifar"] = "imagenet",
    normalization_layer: nn.Module = nn.BatchNorm2d,
) -> _MIMOResNet:
    return _MIMOResNet(
        block=_Bottleneck,
        num_blocks=[3, 8, 36, 3],
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        conv_bias=conv_bias,
        dropout_rate=dropout_rate,
        groups=groups,
        style=style,
        in_planes=64,
        normalization_layer=normalization_layer,
    )
