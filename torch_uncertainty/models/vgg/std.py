# fmt: off
from typing import Type

from torch import nn

from .base import VGG, _vgg
from .configs import cfgs

# fmt:on
__all__ = ["vgg11", "vgg13", "vgg16", "vgg19"]


def vgg11(
    in_channels: int,
    num_classes: int,
    norm: Type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout: float = 0.5,
    style: str = "imagenet",
) -> VGG:
    return _vgg(
        cfgs["A"],
        in_channels=in_channels,
        num_classes=num_classes,
        norm=norm,
        groups=groups,
        dropout=dropout,
        style=style,
    )


def vgg13(
    in_channels: int,
    num_classes: int,
    norm: Type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout: float = 0.5,
    style: str = "imagenet",
) -> VGG:
    return _vgg(
        cfgs["B"],
        in_channels=in_channels,
        num_classes=num_classes,
        norm=norm,
        groups=groups,
        dropout=dropout,
        style=style,
    )


def vgg16(
    in_channels: int,
    num_classes: int,
    norm: Type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout: float = 0.5,
    style: str = "imagenet",
) -> VGG:
    return _vgg(
        cfgs["D"],
        in_channels=in_channels,
        num_classes=num_classes,
        norm=norm,
        groups=groups,
        dropout=dropout,
        style=style,
    )


def vgg19(
    in_channels: int,
    num_classes: int,
    norm: Type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout: float = 0.5,
    style: str = "imagenet",
) -> VGG:
    return _vgg(
        cfgs["E"],
        in_channels=in_channels,
        num_classes=num_classes,
        norm=norm,
        groups=groups,
        dropout=dropout,
        style=style,
    )
