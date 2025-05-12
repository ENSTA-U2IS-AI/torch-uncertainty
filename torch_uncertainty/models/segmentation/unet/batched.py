import torch
from torch import Tensor, nn
from torchvision.transforms import functional as F
from torchvision.transforms import v2

from torch_uncertainty.layers import BatchConv2d, BatchConvTranspose2d


def check_unet_parameters(
    in_channels: int,
    num_classes: int,
    num_blocks: list[int],
    bilinear: bool,
) -> None:
    """Check the parameters for the U-Net model.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        num_blocks (list[int]): Number of channels in each layer of the U-Net.
        bilinear (bool): Whether to use bilinear interpolation for upsampling.
    """
    if len(num_blocks) != 5:
        raise ValueError(f"num_blocks must be a list of 5 integers. Got {len(num_blocks)} blocks.")
    if not all(isinstance(i, int) for i in num_blocks):
        raise ValueError(f"num_blocks must be a list of 5 integers. Got {num_blocks}.")
    if not all(i > 0 for i in num_blocks):
        raise ValueError(f"num_blocks must be a list of 5 positive integers. Got {num_blocks}.")
    if not isinstance(in_channels, int):
        raise TypeError(f"in_channels must be an integer. Got {in_channels}.")
    if not isinstance(num_classes, int):
        raise TypeError(f"num_classes must be an integer. Got {num_classes}.")
    if not isinstance(bilinear, bool):
        raise TypeError(f"bilinear must be a boolean. Got {bilinear}.")
    if in_channels <= 0:
        raise ValueError(f"in_channels must be a positive integer. Got {in_channels}.")
    if num_classes <= 0:
        raise ValueError(f"num_classes must be a positive integer. Got {num_classes}.")


class _DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_estimators: int) -> None:
        """Initialize the DoubleConv module: (Conv2d => BN => ReLU) * 2.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_estimators (int): Number of estimators.
        """
        super().__init__()
        self.conv_block = nn.Sequential(
            BatchConv2d(in_channels, out_channels, 3, num_estimators, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            BatchConv2d(out_channels, out_channels, 3, num_estimators, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.conv_block(x)


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_estimators: int) -> None:
        """Initialize the Down module: (Conv2d => BN => ReLU) * 2 + MaxPool2d.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_estimators (int): Number of estimators.
        """
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            _DoubleConv(in_channels, out_channels, num_estimators),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mpconv(x)


class _Up(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, num_estimators: int, bilinear: bool = True
    ) -> None:
        super().__init__()
        self.bilinear = bilinear

        self.up = BatchConvTranspose2d(
            in_channels // 2, in_channels // 2, 2, num_estimators, stride=2
        )

        self.conv = _DoubleConv(in_channels, out_channels, num_estimators)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        if self.bilinear:
            x1 = F.resize(
                x1,
                size=[2 * x1.size()[2], 2 * x1.size()[3]],
                interpolation=v2.InterpolationMode.BILINEAR,
            )
        else:
            x1 = self.up(x1)
        return self.conv(torch.cat([x2, x1], dim=1))


class _OutputConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_estimators: int) -> None:
        super().__init__()
        self.conv = BatchConv2d(in_channels, out_channels, 1, num_estimators)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _BatchedUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_blocks: list[int],
        num_estimators: int,
        bilinear: bool = False,
    ) -> None:
        """Batched U-Net model using Batch layers for uncertainty estimation.

        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
            num_blocks (list[int]): Number of channels in each layer of the U-Net.
            num_estimators (int): Number of estimators.
            bilinear (bool, optional): If ``True``, use bilinear interpolation instead of
                transposed convolutions for upsampling. Defaults to ``False``.
        """
        check_unet_parameters(in_channels, num_classes, num_blocks, bilinear)
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.bilinear = bilinear

        self.inc = _DoubleConv(in_channels, num_blocks[0], num_estimators=num_estimators)
        self.down1 = _Down(num_blocks[0], num_blocks[1], num_estimators=num_estimators)
        self.down2 = _Down(num_blocks[1], num_blocks[2], num_estimators=num_estimators)
        self.down3 = _Down(num_blocks[2], num_blocks[3], num_estimators=num_estimators)
        factor = 2 if bilinear else 1
        self.down4 = _Down(num_blocks[3], num_blocks[4] // factor, num_estimators=num_estimators)
        self.up1 = _Up(
            num_blocks[4], num_blocks[3] // factor, num_estimators=num_estimators, bilinear=bilinear
        )
        self.up2 = _Up(
            num_blocks[3], num_blocks[2] // factor, num_estimators=num_estimators, bilinear=bilinear
        )
        self.up3 = _Up(
            num_blocks[2], num_blocks[1] // factor, num_estimators=num_estimators, bilinear=bilinear
        )
        self.up4 = _Up(
            num_blocks[1], num_blocks[0], num_estimators=num_estimators, bilinear=bilinear
        )
        self.outc = _OutputConv(num_blocks[0], num_classes, num_estimators=num_estimators)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def _batched_unet(
    in_channels: int,
    num_classes: int,
    num_blocks: list[int],
    num_estimators: int,
    bilinear: bool = False,
) -> _BatchedUNet:
    """Create a Batched U-Net model.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        num_blocks (list[int]): Number of channels in each layer of the U-Net.
        num_estimators (int): Number of estimators.
        bilinear (bool, optional): If ``True``, use bilinear interpolation instead of
            transposed convolutions for upsampling. Defaults to ``False``.

    Returns:
        _BatchedUNet: Batched U-Net model.
    """
    return _BatchedUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        num_blocks=num_blocks,
        num_estimators=num_estimators,
        bilinear=bilinear,
    )


def batched_small_unet(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    bilinear: bool = False,
) -> _BatchedUNet:
    """Create a Small Batched U-Net model (channels divided by 2).

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        num_estimators (int): Number of estimators.
        bilinear (bool, optional): If ``True``, use bilinear interpolation instead of
            transposed convolutions for upsampling. Defaults to ``False``.

    Returns:
        _BatchedUNet: Small Batched U-Net model.
    """
    num_blocks = [32, 64, 128, 256, 512]
    return _batched_unet(
        in_channels=in_channels,
        num_classes=num_classes,
        num_blocks=num_blocks,
        num_estimators=num_estimators,
        bilinear=bilinear,
    )


def batched_unet(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    bilinear: bool = False,
) -> _BatchedUNet:
    """Create a Batched U-Net model.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        num_estimators (int): Number of estimators.
        bilinear (bool, optional): If ``True``, use bilinear interpolation instead of
            transposed convolutions for upsampling. Defaults to ``False``.

    Returns:
        _BatchedUNet: Batched U-Net model.
    """
    num_blocks = [64, 128, 256, 512, 1024]
    return _batched_unet(
        in_channels=in_channels,
        num_classes=num_classes,
        num_blocks=num_blocks,
        num_estimators=num_estimators,
        bilinear=bilinear,
    )
