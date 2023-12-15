from torch import nn

from .base import VGG, _vgg
from .configs import cfgs

__all__ = ["vgg11", "vgg13", "vgg16", "vgg19"]


def vgg11(
    in_channels: int,
    num_classes: int,
    norm: type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout_rate: float = 0.5,
    style: str = "imagenet",
    num_estimators: int | None = None,
) -> VGG:
    return _vgg(
        cfgs["A"],
        in_channels=in_channels,
        num_classes=num_classes,
        norm=norm,
        groups=groups,
        dropout_rate=dropout_rate,
        style=style,
        num_estimators=num_estimators,
    )


def vgg13(
    in_channels: int,
    num_classes: int,
    norm: type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout_rate: float = 0.5,
    style: str = "imagenet",
    num_estimators: int | None = None,
) -> VGG:  # coverage: ignore
    return _vgg(
        cfgs["B"],
        in_channels=in_channels,
        num_classes=num_classes,
        norm=norm,
        groups=groups,
        dropout_rate=dropout_rate,
        style=style,
        num_estimators=num_estimators,
    )


def vgg16(
    in_channels: int,
    num_classes: int,
    norm: type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout_rate: float = 0.5,
    style: str = "imagenet",
    num_estimators: int | None = None,
) -> VGG:  # coverage: ignore
    return _vgg(
        cfgs["D"],
        in_channels=in_channels,
        num_classes=num_classes,
        norm=norm,
        groups=groups,
        dropout_rate=dropout_rate,
        style=style,
        num_estimators=num_estimators,
    )


def vgg19(
    in_channels: int,
    num_classes: int,
    norm: type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout_rate: float = 0.5,
    style: str = "imagenet",
    num_estimators: int | None = None,
) -> VGG:  # coverage: ignore
    return _vgg(
        cfgs["E"],
        in_channels=in_channels,
        num_classes=num_classes,
        norm=norm,
        groups=groups,
        dropout_rate=dropout_rate,
        style=style,
        num_estimators=num_estimators,
    )
