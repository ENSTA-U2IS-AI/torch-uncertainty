from torch import nn

from torch_uncertainty.layers.packed import PackedConv2d, PackedLinear

from .base import VGG, _vgg
from .configs import cfgs

__all__ = [
    "packed_vgg",
]


def packed_vgg(
    in_channels: int,
    num_classes: int,
    arch: int,
    alpha: int,
    num_estimators: int,
    gamma: int,
    norm: type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout_rate: float = 0.5,
    style: str = "imagenet",
) -> VGG:
    if arch == 11:
        config = cfgs["A"]
    elif arch == 13:  # coverage: ignore
        config = cfgs["B"]
    elif arch == 16:  # coverage: ignore
        config = cfgs["D"]
    elif arch == 19:  # coverage: ignore
        config = cfgs["E"]
    else:
        raise ValueError(f"Unknown VGG arch {arch}.")
    return _vgg(
        vgg_cfg=config,
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
