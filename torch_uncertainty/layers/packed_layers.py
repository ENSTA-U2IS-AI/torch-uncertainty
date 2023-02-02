# fmt: off
from typing import Any, Union

import torch.nn as nn
from einops import rearrange
from torch import Tensor
from torch.nn.common_types import _size_2_t


# fmt: on
class PackedLinear(nn.Module):
    """TODO: _summary_

    Args:
        in_features (int): _description_
        out_features (int): _description_
        num_estimators (int): _description_
        bias (bool, optional): _description_. Defaults to True.
        groups (int, optional): _description_. Defaults to 1.
        rearrange (bool, optional): _description_. Defaults to True.
        device (_type_, optional): _description_. Defaults to None.
        dtype (_type_, optional): _description_. Defaults to None.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_estimators: int,
        bias: bool = True,
        groups: int = 1,
        rearrange: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.num_estimators = num_estimators
        self.rearrange = rearrange

        # fix if not divisible by groups
        if in_features % (num_estimators * groups):
            in_features += num_estimators - in_features % (
                num_estimators * groups
            )
        if out_features % (num_estimators * groups):
            out_features += num_estimators - out_features % (
                num_estimators * groups
            )

        self.conv1x1 = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=num_estimators * groups,
            bias=bias,
            padding_mode="zeros",
            **factory_kwargs,
        )

    def _rearrange_forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1).unsqueeze(-1)
        x = rearrange(x, "(m e) c h w -> e (m c) h w", m=self.num_estimators)
        x = self.conv1x1(x)
        x = rearrange(x, "e (m c) h w -> (m e) c h w", m=self.num_estimators)
        return x.squeeze(-1).squeeze(-1)

    def forward(self, input: Tensor) -> Tensor:
        if self.rearrange:
            return self._rearrange_forward(input)
        else:
            return self.conv1x1(input)


class PackedConv2d(nn.Module):
    r"""FIXME: Applies :attr:`num_estimators` 2D convolutions over an input
    signal composed of several input planes using grouped convolutions.

    Args:
        in_channels (int): Number of channels in the input image.
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel.
        n_estimators (int): Number of estimators in the ensemble.
        stride (int or tuple, optional): Stride of the convolution.
            Defaults to ``1``.
        padding (int, tuple or str, optional): Padding added to all four sides
            of the input. Defaults to ``0``.
        dilation (int or tuple, optional): Spacing between kernel elements.
            Defaults to ``1``.
        groups (int, optional): Number of blocked connexions from input
            channels to output channels for each estimator. Defaults to ``1``.
        bias (bool, optional): If ``True``, adds a learnable bias to the
            output. Defaults to ``True``.

    .. note:
        Each ensemble member will only see
        :math:`\frac{in_channels}{n_estimators}` channels, so when using
        :attr:`groups` you have to make sure that :attr:`in_channels` and
        :attr:`out_channels` are both divisible by :attr:`n_estimators`
        :math:`\times`:attr:`groups`.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        num_estimators: int,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        minimum_channels_per_group: int = 64,
        bias: bool = True,
        device: Union[Any, None] = None,
        dtype: Union[Any, None] = None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.num_estimators = num_estimators

        while (
            in_channels % (num_estimators * groups) != 0
            or in_channels // (num_estimators * groups)
            < minimum_channels_per_group
        ) and groups > 1:
            groups -= 1

        # fix if not divisible by groups
        if in_channels % (num_estimators * groups):
            in_channels += (
                num_estimators - in_channels % num_estimators * groups
            )
        if out_channels % (num_estimators * groups):
            out_channels += (
                num_estimators - out_channels % num_estimators * groups
            )

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=num_estimators * groups,
            bias=bias,
            padding_mode="zeros",
            **factory_kwargs,
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.conv(input)
