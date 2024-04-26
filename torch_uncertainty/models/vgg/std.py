from torch import nn

from .base import VGG, _vgg
from .configs import cfgs

__all__ = ["vgg"]


def vgg(
    in_channels: int,
    num_classes: int,
    arch: int,
    norm: type[nn.Module] = nn.Identity,
    groups: int = 1,
    dropout_rate: float = 0.5,
    style: str = "imagenet",
    num_estimators: int | None = None,
) -> VGG:
    if arch == 11:  # coverage: ignore
        config = cfgs["A"]
    elif arch == 13:
        config = cfgs["B"]
    elif arch == 16:
        config = cfgs["D"]
    elif arch == 19:
        config = cfgs["E"]
    else:
        raise ValueError(f"Unknown VGG arch {arch}.")
    return _vgg(
        vgg_cfg=config,
        in_channels=in_channels,
        num_classes=num_classes,
        norm=norm,
        groups=groups,
        dropout_rate=dropout_rate,
        style=style,
        num_estimators=num_estimators,
    )
