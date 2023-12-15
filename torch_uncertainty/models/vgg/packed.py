from torch import nn

from torch_uncertainty.layers.packed import PackedConv2d, PackedLinear

from .base import VGG, _vgg
from .configs import cfgs

__all__ = [
    "packed_vgg11",
    "packed_vgg13",
    "packed_vgg16",
    "packed_vgg19",
]


def packed_vgg11(
    in_channels: int,
    num_classes: int,
    alpha: int,
    num_estimators: int,
    gamma: int,
    norm: type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout_rate: float = 0.5,
    style: str = "imagenet",
) -> VGG:
    return _vgg(
        cfgs["A"],
        in_channels=in_channels,
        num_classes=num_classes,
        linear_layer=PackedLinear,
        conv2d_layer=PackedConv2d,
        norm=norm,
        groups=groups,
        dropout_rate=dropout_rate,
        style=style,
        alpha=alpha,
        num_estimators=num_estimators,
        gamma=gamma,
    )


def packed_vgg13(
    in_channels: int,
    num_classes: int,
    alpha: int,
    num_estimators: int,
    gamma: int,
    norm: type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout_rate: float = 0.5,
    style: str = "imagenet",
) -> VGG:  # coverage: ignore
    return _vgg(
        cfgs["B"],
        in_channels=in_channels,
        num_classes=num_classes,
        linear_layer=PackedLinear,
        conv2d_layer=PackedConv2d,
        norm=norm,
        groups=groups,
        dropout_rate=dropout_rate,
        style=style,
        alpha=alpha,
        num_estimators=num_estimators,
        gamma=gamma,
    )


def packed_vgg16(
    in_channels: int,
    num_classes: int,
    alpha: int,
    num_estimators: int,
    gamma: int,
    norm: type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout_rate: float = 0.5,
    style: str = "imagenet",
) -> VGG:  # coverage: ignore
    return _vgg(
        cfgs["D"],
        in_channels=in_channels,
        num_classes=num_classes,
        linear_layer=PackedLinear,
        conv2d_layer=PackedConv2d,
        norm=norm,
        groups=groups,
        dropout_rate=dropout_rate,
        style=style,
        alpha=alpha,
        num_estimators=num_estimators,
        gamma=gamma,
    )


def packed_vgg19(
    in_channels: int,
    num_classes: int,
    alpha: int,
    num_estimators: int,
    gamma: int,
    norm: type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout_rate: float = 0.5,
    style: str = "imagenet",
) -> VGG:  # coverage: ignore
    return _vgg(
        cfgs["E"],
        in_channels=in_channels,
        num_classes=num_classes,
        linear_layer=PackedLinear,
        conv2d_layer=PackedConv2d,
        norm=norm,
        groups=groups,
        dropout_rate=dropout_rate,
        style=style,
        alpha=alpha,
        num_estimators=num_estimators,
        gamma=gamma,
    )
