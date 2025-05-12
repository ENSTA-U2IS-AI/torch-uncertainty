from einops import rearrange
from torch import Tensor

from .standard import _UNet


class _MIMOUNet(_UNet):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_blocks: list[int],
        num_estimators: int,
        bilinear: bool = False,
    ):
        super().__init__(
            in_channels * num_estimators, num_classes * num_estimators, num_blocks, bilinear
        )
        self.num_estimators = num_estimators

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            x = x.repeat(self.num_estimators, 1, 1, 1)
        out = rearrange(x, "(m b) c ... -> b (m c) ...", m=self.num_estimators)
        out = super().forward(out)
        return rearrange(out, "b (m c) ... -> (m b) c ...", m=self.num_estimators)


def _mimo_unet(
    in_channels: int,
    num_classes: int,
    num_blocks: list[int],
    num_estimators: int,
    bilinear: bool = False,
) -> _MIMOUNet:
    return _MIMOUNet(in_channels, num_classes, num_blocks, num_estimators, bilinear)


def mimo_small_unet(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    bilinear: bool = False,
) -> _MIMOUNet:
    num_blocks = [32, 64, 128, 256, 512]
    return _mimo_unet(in_channels, num_classes, num_blocks, num_estimators, bilinear)


def mimo_unet(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    bilinear: bool = False,
) -> _MIMOUNet:
    num_blocks = [64, 128, 256, 512, 1024]
    return _mimo_unet(in_channels, num_classes, num_blocks, num_estimators, bilinear)
