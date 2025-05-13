import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn
from torch.nn.common_types import _size_2_t
from torchvision.transforms import v2

from torch_uncertainty.layers.packed import PackedConv2d, check_packed_parameters_consistency


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


class PackedConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        alpha: int,
        num_estimators: int,
        gamma: int = 1,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        minimum_channels_per_group: int = 64,
        bias: bool = True,
        first: bool = False,
        last: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        r"""Packed-Ensembles-style ConvTranspose2d layer with debug flags.

        Args:
            in_channels (int): Number of channels in the input.
            out_channels (int): Number of channels produced by the transposed convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            alpha (int): The channel multiplier for the layer.
            num_estimators (int): Number of estimators in the ensemble.
            gamma (int, optional): Defaults to ``1``.
            stride (int or tuple, optional): Stride of the convolution. Defaults to ``1``.
            padding (int or tuple, optional): Zero-padding added to both sides of the input. Defaults to ``0``.
            output_padding (int or tuple, optional): Additional size added to one side of the output shape. Defaults to ``0``.
            dilation (int or tuple, optional): Spacing between kernel elements. Defaults to ``1``.
            groups (int, optional): Number of blocked connections from input channels to output channels. Defaults to ``1``.
            minimum_channels_per_group (int, optional): Smallest possible number of channels per group.
            bias (bool, optional): If ``True``, adds a learnable bias to the output. Defaults to ``True``.
            first (bool, optional): Whether this is the first layer of the network. Defaults to ``False``.
            last (bool, optional): Whether this is the last layer of the network. Defaults to ``False``.
            device (torch.device, optional): The device to use for the layer's parameters. Defaults to ``None``.
            dtype (torch.dtype, optional): The dtype to use for the layer's parameters. Defaults to ``None``.
        """
        check_packed_parameters_consistency(alpha, gamma, num_estimators)
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.num_estimators = num_estimators
        self.first = first
        self.last = last

        # Define the number of channels for the underlying convolution
        self.extended_in_channels = int(in_channels * (1 if first else alpha))
        self.extended_out_channels = int(out_channels * (num_estimators if last else alpha))

        # Define the number of groups of the underlying convolution
        self.actual_groups = 1 if first else gamma * groups * num_estimators

        while (
            self.extended_in_channels % self.actual_groups != 0
            or self.extended_in_channels // self.actual_groups < minimum_channels_per_group
        ) and self.actual_groups // (groups * num_estimators) > 1:
            gamma -= 1
            self.actual_groups = gamma * groups * num_estimators

        # Fix dimensions to be divisible by groups
        if self.extended_in_channels % self.actual_groups:
            self.extended_in_channels += (
                num_estimators - self.extended_in_channels % self.actual_groups
            )
        if self.extended_out_channels % self.actual_groups:
            self.extended_out_channels += (
                num_estimators - self.extended_out_channels % self.actual_groups
            )

        # Initialize the transposed convolutional layer
        self.conv_transpose = nn.ConvTranspose2d(
            in_channels=self.extended_in_channels,
            out_channels=self.extended_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=self.actual_groups,
            bias=bias,
            **factory_kwargs,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv_transpose(inputs)

    @property
    def weight(self) -> Tensor:
        r"""The weight of the underlying transposed convolutional layer."""
        return self.conv_transpose.weight

    @property
    def bias(self) -> Tensor | None:
        r"""The bias of the underlying transposed convolutional layer."""
        return self.conv_transpose.bias


class PackedDoubleConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        alpha: float,
        num_estimators: int,
        gamma: int,
        first: bool = False,
        last: bool = False,
    ) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            PackedConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                alpha=alpha,
                num_estimators=num_estimators,
                gamma=gamma,
                padding=1,
                first=first,
            ),
            nn.BatchNorm2d(num_features=int(out_channels * (num_estimators if last else alpha))),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class PackedInconv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, alpha: float, num_estimators: int, gamma: int
    ) -> None:
        super().__init__()
        self.conv = PackedDoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            first=True,
        )

    def forward(self, x):
        return self.conv(x)


class PackedDown(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, alpha: float, num_estimators: int, gamma: int
    ) -> None:
        super().__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            PackedDoubleConv(
                in_channels=in_channels,
                out_channels=out_channels,
                alpha=alpha,
                num_estimators=num_estimators,
                gamma=gamma,
            ),
        )

    def forward(self, x):
        return self.mpconv(x)


class PackedUp(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        alpha: float,
        num_estimators: int,
        gamma: int,
        bilinear: bool = True,
    ) -> None:
        super().__init__()
        self.bilinear = bilinear
        self.num_estimators = num_estimators
        self.up = PackedConvTranspose2d(
            in_channels=in_channels // 2,
            out_channels=in_channels // 2,
            kernel_size=2,
            stride=2,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
        )
        self.conv = PackedDoubleConv(
            in_channels=in_channels,
            out_channels=out_channels,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
        )

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = F.resize(
                x1,
                size=[2 * x1.size()[2], 2 * x1.size()[3]],
                interpolation=v2.InterpolationMode.BILINEAR,
            )
        else:
            x1 = self.up(x1)

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x1 = rearrange(x1, "b (m c) h w -> b m c h w", m=self.num_estimators)
        x2 = rearrange(x2, "b (m c) h w -> b m c h w", m=self.num_estimators)
        x = torch.cat([x2, x1], dim=2)
        x = rearrange(x, "b m c h w -> b (m c) h w")
        return self.conv(x)


class PackedOutconv(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, alpha: float, num_estimators: int, gamma: int
    ) -> None:
        super().__init__()
        self.conv = PackedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            last=True,
        )

    def forward(self, x):
        return self.conv(x)


class _PackedUNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_blocks: list[int],
        alpha: float = 1,
        num_estimators: int = 1,
        gamma: int = 1,
        bilinear: bool = False,
    ) -> None:
        check_unet_parameters(in_channels, num_classes, num_blocks, bilinear)
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.alpha = alpha
        self.num_estimators = num_estimators
        self.gamma = gamma
        self.bilinear = bilinear

        # Downsampling
        self.inc = PackedInconv(in_channels, num_blocks[0], alpha, num_estimators, gamma)
        self.down1 = PackedDown(num_blocks[0], num_blocks[1], alpha, num_estimators, gamma)
        self.down2 = PackedDown(num_blocks[1], num_blocks[2], alpha, num_estimators, gamma)
        self.down3 = PackedDown(num_blocks[2], num_blocks[3], alpha, num_estimators, gamma)
        factor = 2 if bilinear else 1
        self.down4 = PackedDown(
            num_blocks[3], num_blocks[4] // factor, alpha, num_estimators, gamma
        )
        # Upsampling
        self.up1 = PackedUp(
            num_blocks[4], num_blocks[3] // factor, alpha, num_estimators, gamma, bilinear
        )
        self.up2 = PackedUp(
            num_blocks[3], num_blocks[2] // factor, alpha, num_estimators, gamma, bilinear
        )
        self.up3 = PackedUp(
            num_blocks[2], num_blocks[1] // factor, alpha, num_estimators, gamma, bilinear
        )
        self.up4 = PackedUp(
            num_blocks[1], num_blocks[0] // factor, alpha, num_estimators, gamma, bilinear
        )

        # Dropout
        self.dropout = nn.Dropout2d(0.1)

        # Final output
        self.outc = PackedOutconv(num_blocks[0], num_classes, alpha, num_estimators, gamma)

    def forward(self, x):
        # Downsampling
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        # Upsampling
        x = self.up1(x5, x4)
        x = self.dropout(x)
        x = self.up2(x, x3)
        x = self.dropout(x)
        x = self.up3(x, x2)
        x = self.dropout(x)
        x = self.up4(x, x1)
        x = self.dropout(x)

        # Final output
        return self.outc(x)


def _packed_unet(
    in_channels: int,
    num_classes: int,
    num_blocks: list[int],
    bilinear: bool = False,
    alpha: float = 1,
    num_estimators: int = 1,
    gamma: int = 1,
) -> _PackedUNet:
    """_summary_.

    Args:
        in_channels (int): _description_
        num_classes (int): _description_
        num_blocks (list[int]): _description_
        bilinear (bool, optional): _description_. Defaults to False.
        alpha (float, optional): _description_. Defaults to 1.
        num_estimators (int, optional): _description_. Defaults to 1.
        gamma (int, optional): _description_. Defaults to 1.

    Returns:
        PackedUNet: _description_
    """
    return _PackedUNet(
        in_channels=in_channels,
        num_classes=num_classes,
        num_blocks=num_blocks,
        alpha=alpha,
        num_estimators=num_estimators,
        gamma=gamma,
        bilinear=bilinear,
    )


def packed_small_unet(
    in_channels: int,
    num_classes: int,
    bilinear: bool = False,
    alpha: float = 1,
    num_estimators: int = 1,
    gamma: int = 1,
) -> _PackedUNet:
    """Create a Packed-Ensembles of small U-Net models.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        bilinear (bool, optional): If ``True``, use bilinear interpolation instead of
            transposed convolutions for upsampling. This can help to reduce the number
            of parameters and improve the performance of the model. Defaults to ``False``.
        alpha (float, optional): _description_. Defaults to 1.
        num_estimators (int, optional): _description_. Defaults to 1.
        gamma (int, optional): _description_. Defaults to 1.

    Returns:
        PackedUNet: U-Net model.
    """
    return _packed_unet(
        in_channels=in_channels,
        num_classes=num_classes,
        num_blocks=[32, 64, 128, 256, 512],
        bilinear=bilinear,
        alpha=alpha,
        num_estimators=num_estimators,
        gamma=gamma,
    )


def packed_unet(
    in_channels: int,
    num_classes: int,
    bilinear: bool = False,
    alpha: float = 1,
    num_estimators: int = 1,
    gamma: int = 1,
) -> _PackedUNet:
    """Create a Packed-Ensembles of U-Net models.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        bilinear (bool, optional): If ``True``, use bilinear interpolation instead
            of transposed convolutions for upsampling. This can help to reduce the number
            of parameters and improve the performance of the model. Defaults to ``False``.
        alpha (float, optional): _description_. Defaults to 1.
        num_estimators (int, optional): _description_. Defaults to 1.
        gamma (int, optional): _description_. Defaults to 1.

    Returns:
        PackedUNet: U-Net model.
    """
    return _packed_unet(
        in_channels=in_channels,
        num_classes=num_classes,
        num_blocks=[64, 128, 256, 512, 1024],
        bilinear=bilinear,
        alpha=alpha,
        num_estimators=num_estimators,
        gamma=gamma,
    )
