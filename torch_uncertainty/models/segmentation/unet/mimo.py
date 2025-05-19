from einops import rearrange, repeat
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
        dropout_rate: float = 0.0,
    ):
        super().__init__(
            in_channels=in_channels * num_estimators,
            num_classes=num_classes * num_estimators,
            num_blocks=num_blocks,
            bilinear=bilinear,
            dropout_rate=dropout_rate,
        )
        self.num_estimators = num_estimators

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            x = repeat(x, "b ... -> (m b) ...", m=self.num_estimators)
        out = rearrange(x, "(m b) c ... -> b (m c) ...", m=self.num_estimators)
        out = super().forward(out)
        return rearrange(out, "b (m c) ... -> (m b) c ...", m=self.num_estimators)


def _mimo_unet(
    in_channels: int,
    num_classes: int,
    num_blocks: list[int],
    num_estimators: int,
    bilinear: bool = False,
    dropout_rate: float = 0.0,
) -> _MIMOUNet:
    return _MIMOUNet(
        in_channels, num_classes, num_blocks, num_estimators, bilinear, dropout_rate=dropout_rate
    )


def mimo_small_unet(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    bilinear: bool = False,
    dropout_rate: float = 0.0,
) -> _MIMOUNet:
    num_blocks = [32, 64, 128, 256, 512]
    return _mimo_unet(
        in_channels, num_classes, num_blocks, num_estimators, bilinear, dropout_rate=dropout_rate
    )


def mimo_unet(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    bilinear: bool = False,
    dropout_rate: float = 0.0,
) -> _MIMOUNet:
    num_blocks = [64, 128, 256, 512, 1024]
    return _mimo_unet(
        in_channels, num_classes, num_blocks, num_estimators, bilinear, dropout_rate=dropout_rate
    )
