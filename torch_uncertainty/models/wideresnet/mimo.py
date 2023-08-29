# fmt: off
import torch
from einops import rearrange

from .std import _Wide

# fmt: on
__all__ = [
    "mimo_wideresnet28x10",
]


class _MIMOWide(_Wide):
    def __init__(
        self,
        depth: int,
        widen_factor: int,
        in_channels: int,
        num_classes: int,
        num_estimators: int,
        dropout_rate: float,
        groups: int = 1,
        style: str = "imagenet",
    ):
        super().__init__(
            depth,
            widen_factor=widen_factor,
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


def mimo_wideresnet28x10(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    groups: int = 1,
    style: str = "imagenet",
) -> _MIMOWide:
    return _MIMOWide(
        depth=28,
        widen_factor=10,
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        dropout_rate=0.3,
        groups=groups,
        style=style,
    )
