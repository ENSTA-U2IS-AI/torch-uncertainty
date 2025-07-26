# Code from https://github.com/milesial/Pytorch-UNet with slight modifications

import torch
from torch import Tensor, nn
from torchvision.transforms import functional as F


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
    """(convolution => [BN] => ReLU) * 2."""

    def __init__(
        self, in_channels: int, out_channels: int, mid_channels: int | None = None
    ) -> None:
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=mid_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_features=out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.double_conv(x)


class _Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.mpconv = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(in_channels, out_channels))

    def forward(self, x: Tensor) -> Tensor:
        return self.mpconv(x)


class _Up(nn.Module):
    """Upscaling then double conv."""

    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True) -> None:
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = _DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = _DoubleConv(in_channels, out_channels)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up(x1)
        # input is CHW
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_y // 2, diff_x - diff_x // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class _OutConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class _UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_blocks: list[int],
        bilinear: bool = False,
        dropout_rate: float = 0.0,
    ) -> None:
        """U-Net model from the paper "U-Net: Convolutional Networks for Biomedical Image Segmentation"
        (Ronneberger et al., 2015). This implementation is based on the original paper and is
        designed for image segmentation tasks. The U-Net architecture consists of a contracting path
        (encoder) and an expansive path (decoder).

        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
            num_blocks (list[int]): Number of channels in each layer of the U-Net.
            bilinear (bool, optional): If ``True``, use bilinear interpolation instead of
                transposed convolutions for upsampling. This can help to reduce the number
                of parameters and improve the performance of the model. Defaults to ``False``.
            dropout_rate (float, optional): Dropout rate for the model. Defaults to 0.0.
        """
        check_unet_parameters(in_channels, num_classes, num_blocks, bilinear)
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.bilinear = bilinear

        self.inc = _DoubleConv(in_channels, num_blocks[0])
        self.down1 = _Down(num_blocks[0], num_blocks[1])
        self.down2 = _Down(num_blocks[1], num_blocks[2])
        self.down3 = _Down(num_blocks[2], num_blocks[3])
        factor = 2 if bilinear else 1
        self.down4 = _Down(num_blocks[3], num_blocks[4] // factor)
        self.up1 = _Up(num_blocks[4], num_blocks[3] // factor, bilinear)
        self.up2 = _Up(num_blocks[3], num_blocks[2] // factor, bilinear)
        self.up3 = _Up(num_blocks[2], num_blocks[1] // factor, bilinear)
        self.up4 = _Up(num_blocks[1], num_blocks[0], bilinear)
        self.outc = _OutConv(num_blocks[0], num_classes)

        # Dropout
        self.dropout = nn.Dropout2d(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        # Downsampling
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.dropout(x2)
        x3 = self.down2(x2)
        x3 = self.dropout(x3)
        x4 = self.down3(x3)
        x4 = self.dropout(x4)
        x5 = self.down4(x4)
        # Upsampling
        x5 = self.dropout(x5)
        x = self.up1(x5, x4)
        x = self.dropout(x)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        # Final output
        return self.outc(x)


def _unet(
    in_channels: int,
    num_classes: int,
    num_blocks: list[int],
    bilinear: bool = False,
    dropout_rate: float = 0.0,
) -> _UNet:
    """Create a U-Net model.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        num_blocks (list[int]): Number of channels in each layer of the U-Net.
        bilinear (bool, optional): If ``True``, use bilinear interpolation instead of
            transposed convolutions for upsampling. This can help to reduce the number
            of parameters and improve the performance of the model. Defaults to ``False``.
        dropout_rate (float, optional): Dropout rate for the model. Defaults to 0.0.

    Returns:
        _UNet: U-Net model.
    """
    return _UNet(
        in_channels=in_channels,
        num_classes=num_classes,
        num_blocks=num_blocks,
        bilinear=bilinear,
        dropout_rate=dropout_rate,
    )


def small_unet(
    in_channels: int,
    num_classes: int,
    bilinear: bool = False,
    dropout_rate: float = 0.0,
) -> _UNet:
    """Create a Small U-Net model (channels divided by 2).

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        bilinear (bool, optional): If ``True``, use bilinear interpolation instead of
            transposed convolutions for upsampling. This can help to reduce the number
            of parameters and improve the performance of the model. Defaults to ``False``.
        dropout_rate (float, optional): Dropout rate for the model. Defaults to 0.0.

    Returns:
        _SmallUNet: Small U-Net model.
    """
    num_blocks = [32, 64, 128, 256, 512]
    return _unet(
        in_channels=in_channels,
        num_classes=num_classes,
        num_blocks=num_blocks,
        bilinear=bilinear,
        dropout_rate=dropout_rate,
    )


def unet(
    in_channels: int,
    num_classes: int,
    bilinear: bool = False,
    dropout_rate: float = 0.0,
) -> _UNet:
    """Create a U-Net model.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        bilinear (bool, optional): If ``True``, use bilinear interpolation instead of
            transposed convolutions for upsampling. This can help to reduce the number
            of parameters and improve the performance of the model. Defaults to ``False``.
        dropout_rate (float, optional): Dropout rate for the model. Defaults to 0.0.

    Returns:
        _UNet: U-Net model.
    """
    num_blocks = [64, 128, 256, 512, 1024]
    return _unet(
        in_channels=in_channels,
        num_classes=num_classes,
        num_blocks=num_blocks,
        bilinear=bilinear,
        dropout_rate=dropout_rate,
    )
