import torch
from einops import repeat
from torch import Tensor, nn
from torchvision.transforms import functional as F

from torch_uncertainty.layers import BatchConv2d, BatchConvTranspose2d

from .standard import check_unet_parameters


class _DoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_estimators: int,
        mid_channels: int | None = None,
    ) -> None:
        """Initialize the DoubleConv module: (Conv2d => BN => ReLU) * 2.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_estimators (int): Number of estimators.
            mid_channels (int | None, optional): Number of intermediate channels.
                If ``None``, defaults to :attr:`out_channels`. Defaults to ``None``.
        """
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.conv_block = nn.Sequential(
            BatchConv2d(in_channels, mid_channels, 3, num_estimators, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            BatchConv2d(mid_channels, out_channels, 3, num_estimators, padding=1),
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

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = _DoubleConv(
                in_channels,
                out_channels,
                num_estimators=num_estimators,
                mid_channels=in_channels // 2,
            )
        else:
            self.up = self.up = BatchConvTranspose2d(
                in_channels, in_channels // 2, 2, num_estimators, stride=2
            )
            self.conv = _DoubleConv(in_channels, out_channels, num_estimators=num_estimators)

    def forward(self, x1: Tensor, x2: Tensor) -> Tensor:
        x1 = self.up(x1)
        # input is CHW
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_y // 2, diff_x - diff_x // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


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
        dropout_rate: float = 0.0,
    ) -> None:
        """Batched U-Net model using Batch layers for uncertainty estimation.

        Args:
            in_channels (int): Number of input channels.
            num_classes (int): Number of output classes.
            num_blocks (list[int]): Number of channels in each layer of the U-Net.
            num_estimators (int): Number of estimators.
            bilinear (bool, optional): If ``True``, use bilinear interpolation instead of
                transposed convolutions for upsampling. Defaults to ``False``.
            dropout_rate (float, optional): Dropout rate. Defaults to ``0.0``.

        """
        check_unet_parameters(in_channels, num_classes, num_blocks, bilinear)
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.num_estimators = num_estimators
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

        self.dropout = nn.Dropout2d(dropout_rate)

        self.outc = _OutputConv(num_blocks[0], num_classes, num_estimators=num_estimators)

    def forward(self, x: Tensor) -> Tensor:
        x = repeat(x, "b ... -> (m b) ...", m=self.num_estimators)
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


def _batched_unet(
    in_channels: int,
    num_classes: int,
    num_blocks: list[int],
    num_estimators: int,
    bilinear: bool = False,
    dropout_rate: float = 0.0,
) -> _BatchedUNet:
    """Create a Batched U-Net model.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        num_blocks (list[int]): Number of channels in each layer of the U-Net.
        num_estimators (int): Number of estimators.
        bilinear (bool, optional): If ``True``, use bilinear interpolation instead of
            transposed convolutions for upsampling. Defaults to ``False``.
        dropout_rate (float, optional): Dropout rate. Defaults to ``0.0``.

    Returns:
        _BatchedUNet: Batched U-Net model.
    """
    return _BatchedUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        num_blocks=num_blocks,
        num_estimators=num_estimators,
        bilinear=bilinear,
        dropout_rate=dropout_rate,
    )


def batched_small_unet(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    bilinear: bool = False,
    dropout_rate: float = 0.0,
) -> _BatchedUNet:
    """Create a Small Batched U-Net model (channels divided by 2).

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        num_estimators (int): Number of estimators.
        bilinear (bool, optional): If ``True``, use bilinear interpolation instead of
            transposed convolutions for upsampling. Defaults to ``False``.
        dropout_rate (float, optional): Dropout rate. Defaults to ``0.0``.

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
        dropout_rate=dropout_rate,
    )


def batched_unet(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    bilinear: bool = False,
    dropout_rate: float = 0.0,
) -> _BatchedUNet:
    """Create a Batched U-Net model.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        num_estimators (int): Number of estimators.
        bilinear (bool, optional): If ``True``, use bilinear interpolation instead of
            transposed convolutions for upsampling. Defaults to ``False``.
        dropout_rate (float, optional): Dropout rate. Defaults to ``0.0``.

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
        dropout_rate=dropout_rate,
    )
