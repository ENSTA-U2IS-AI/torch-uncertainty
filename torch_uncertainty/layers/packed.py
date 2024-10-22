import math
from typing import Any

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t


def check_packed_parameters_consistency(
    alpha: float, gamma: int, num_estimators: int
) -> None:
    """Check the consistency of the parameters of the Packed-Ensembles layers.

    Args:
        alpha (int): The width multiplier of the layer.
        gamma (int): The number of groups in the ensemble.
        num_estimators (int): The number of estimators in the ensemble.
    """
    if alpha is None:
        raise ValueError("You must specify the value of the arg. `alpha`")

    if alpha <= 0:
        raise ValueError(f"Attribute `alpha` should be > 0, not {alpha}")

    if not isinstance(gamma, int):
        raise TypeError(
            f"Attribute `gamma` should be an int, not {type(gamma)}"
        )
    if gamma <= 0:
        raise ValueError(f"Attribute `gamma` should be >= 1, not {gamma}")

    if num_estimators is None:
        raise ValueError(
            "You must specify the value of the arg. `num_estimators`"
        )
    if not isinstance(num_estimators, int):
        raise TypeError(
            "Attribute `num_estimators` should be an int, not "
            f"{type(num_estimators)}"
        )
    if num_estimators <= 0:
        raise ValueError(
            "Attribute `num_estimators` should be >= 1, not "
            f"{num_estimators}"
        )


class PackedLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        alpha: float,
        num_estimators: int,
        gamma: int = 1,
        bias: bool = True,
        first: bool = False,
        last: bool = False,
        implementation: str = "legacy",
        rearrange: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        r"""Packed-Ensembles-style Linear layer.

        This layer computes fully-connected operation for a given number of
        estimators (:attr:`num_estimators`).

        Args:
            in_features (int): Number of input features of the linear layer.
            out_features (int): Number of channels produced by the linear layer.
            alpha (float): The width multiplier of the linear layer.
            num_estimators (int): The number of estimators grouped in the layer.
            gamma (int, optional): Defaults to ``1``.
            bias (bool, optional): It ``True``, adds a learnable bias to the
                output. Defaults to ``True``.
            first (bool, optional): Whether this is the first layer of the
                network. Defaults to ``False``.
            last (bool, optional): Whether this is the last layer of the network.
                Defaults to ``False``.
            implementation (str, optional): The implementation to use. Defaults
                to ``"legacy"``.
            rearrange (bool, optional): Rearrange the input and outputs for
                compatibility with previous and later layers. Defaults to ``True``.
            device (torch.device, optional): The device to use for the layer's
                parameters. Defaults to ``None``.
            dtype (torch.dtype, optional): The dtype to use for the layer's
                parameters. Defaults to ``None``.

        Explanation Note:
            Increasing :attr:`alpha` will increase the number of channels of the
            ensemble, increasing its representation capacity. Increasing
            :attr:`gamma` will increase the number of groups in the network and
            therefore reduce the number of parameters.

        Note:
            Each ensemble member will only see
            :math:`\frac{\text{in_features}}{\text{num_estimators}}` features,
            so when using :attr:`gamma` you should make sure that
            :attr:`in_features` and :attr:`out_features` are both divisible by
            :attr:`n_estimators` :math:`\times`:attr:`gamma`. However, the
            number of input and output features will be changed to comply with
            this constraint.

        Note:
            The input should be of shape (`batch_size`, :attr:`in_features`, 1,
            1). The (often) necessary rearrange operation is executed by
            default.
        """
        check_packed_parameters_consistency(alpha, gamma, num_estimators)
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.first = first
        self.num_estimators = num_estimators
        self.rearrange = rearrange
        self.implementation = implementation

        # Define the number of features of the underlying convolution
        extended_in_features = int(in_features * (1 if first else alpha))
        extended_out_features = int(
            out_features * (num_estimators if last else alpha)
        )

        # Define the number of groups of the underlying convolution
        actual_groups = num_estimators * gamma if not first else 1

        # fix if not divisible by groups
        if extended_in_features % actual_groups:
            extended_in_features += num_estimators - extended_in_features % (
                actual_groups
            )
        if extended_out_features % actual_groups:
            extended_out_features += num_estimators - extended_out_features % (
                actual_groups
            )

        # FIXME: This is a temporary check
        assert implementation in [
            "legacy",
            "sparse",
            "full",
            "einsum",
        ], f"Unknown implementation: {implementation} for PackedLinear"

        if self.implementation == "legacy":
            self.weight = nn.Parameter(
                torch.empty(
                    (
                        extended_out_features,
                        extended_in_features // actual_groups,
                        1,
                    ),
                    **factory_kwargs,
                )
            )
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    (
                        actual_groups,
                        extended_out_features // actual_groups,
                        extended_in_features // actual_groups,
                    ),
                    **factory_kwargs,
                )
            )

        self.in_features = extended_in_features // actual_groups
        self.out_features = extended_out_features // actual_groups
        self.groups = actual_groups

        if bias:
            self.bias = nn.Parameter(
                torch.empty(extended_out_features, **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.implementation == "legacy":
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            for n in range(self.groups):
                nn.init.kaiming_uniform_(self.weight[n], a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[0])
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

        if self.implementation == "sparse":
            self.weight = nn.Parameter(
                torch.block_diag(*self.weight).to_sparse()
            )

    def _rearrange_forward(self, x: Tensor) -> Tensor:
        x = x.unsqueeze(-1)
        if not self.first:
            x = rearrange(x, "(m e) c h -> e (m c) h", m=self.num_estimators)
        x = F.conv1d(x, self.weight, self.bias, 1, 0, 1, self.groups)
        x = rearrange(x, "e (m c) h -> (m e) c h", m=self.num_estimators)
        return x.squeeze(-1)

    def forward(self, inputs: Tensor) -> Tensor:
        if self.implementation == "legacy":
            if self.rearrange:
                return self._rearrange_forward(inputs)
            return F.conv1d(
                inputs, self.weight, self.bias, 1, 0, 1, self.groups
            )
        if self.implementation == "full":
            block_diag = torch.block_diag(*self.weight)
            return F.linear(inputs, block_diag, self.bias)
        if self.implementation == "sparse":
            return (inputs @ self.weight.transpose(0, 1)) + self.bias
        if self.implementation == "einsum":
            return torch.einsum(
                "bki,kij->bkj",
                inputs.view(-1, self.groups, self.in_features),
                self.weight.transpose(1, 2),
            ).flatten(start_dim=-2, end_dim=-1)
        raise ValueError(f"Unknown implementation: {self.implementation}")


class PackedConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        alpha: int,
        num_estimators: int,
        gamma: int = 1,
        stride: _size_1_t = 1,
        padding: str | _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        minimum_channels_per_group: int = 64,
        bias: bool = True,
        padding_mode: str = "zeros",
        first: bool = False,
        last: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        r"""Packed-Ensembles-style Conv1d layer.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            alpha (int): The channel multiplier of the convolutional layer.
            num_estimators (int): Number of estimators in the ensemble.
            gamma (int, optional): Defaults to ``1``.
            stride (int or tuple, optional): Stride of the convolution.
                Defaults to ``1``.
            padding (int, tuple or str, optional): Padding added to both sides of
                the input. Defaults to ``0``.
            dilation (int or tuple, optional): Spacing between kernel elements.
                Defaults to ``1``.
            groups (int, optional): Number of blocked connexions from input
                channels to output channels for each estimator. Defaults to ``1``.
            minimum_channels_per_group (int, optional): Smallest possible number of
                channels per group.
            bias (bool, optional): If ``True``, adds a learnable bias to the
                output. Defaults to ``True``.
            padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
                ``'replicate'`` or ``'circular'``. Defaults to ``'zeros'``.
            first (bool, optional): Whether this is the first layer of the
                network. Defaults to ``False``.
            last (bool, optional): Whether this is the last layer of the network.
                Defaults to ``False``.
            device (torch.device, optional): The device to use for the layer's
                parameters. Defaults to ``None``.
            dtype (torch.dtype, optional): The dtype to use for the layer's
                parameters. Defaults to ``None``.

        Explanation Note:
            Increasing :attr:`alpha` will increase the number of channels of the
            ensemble, increasing its representation capacity. Increasing
            :attr:`gamma` will increase the number of groups in the network and
            therefore reduce the number of parameters.

        Note:
            Each ensemble member will only see
            :math:`\frac{\text{in_channels}}{\text{num_estimators}}` channels,
            so when using :attr:`groups` you should make sure that
            :attr:`in_channels` and :attr:`out_channels` are both divisible by
            :attr:`num_estimators` :math:`\times`:attr:`gamma` :math:`\times`
            :attr:`groups`. However, the number of input and output channels will
            be changed to comply with this constraint.
        """
        check_packed_parameters_consistency(alpha, gamma, num_estimators)
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.num_estimators = num_estimators

        # Define the number of channels of the underlying convolution
        extended_in_channels = int(in_channels * (1 if first else alpha))
        extended_out_channels = int(
            out_channels * (num_estimators if last else alpha)
        )

        # Define the number of groups of the underlying convolution
        actual_groups = 1 if first else gamma * groups * num_estimators

        while (
            extended_in_channels % actual_groups != 0
            or extended_in_channels // actual_groups
            < minimum_channels_per_group
        ) and actual_groups // (groups * num_estimators) > 1:
            gamma -= 1
            actual_groups = gamma * groups * num_estimators

        # fix if not divisible by groups
        if extended_in_channels % actual_groups:
            extended_in_channels += (
                num_estimators - extended_in_channels % actual_groups
            )
        if extended_out_channels % actual_groups:
            extended_out_channels += (
                num_estimators - extended_out_channels % actual_groups
            )

        self.conv = nn.Conv1d(
            in_channels=extended_in_channels,
            out_channels=extended_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=actual_groups,
            bias=bias,
            padding_mode=padding_mode,
            **factory_kwargs,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

    @property
    def weight(self) -> Tensor:
        r"""The weight of the underlying convolutional layer."""
        return self.conv.weight

    @property
    def bias(self) -> Tensor | None:
        r"""The bias of the underlying convolutional layer."""
        return self.conv.bias


class PackedConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        alpha: int,
        num_estimators: int,
        gamma: int = 1,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        minimum_channels_per_group: int = 64,
        bias: bool = True,
        padding_mode: str = "zeros",
        first: bool = False,
        last: bool = False,
        device: Any | None = None,
        dtype: Any | None = None,
    ) -> None:
        r"""Packed-Ensembles-style Conv2d layer.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            alpha (int): The channel multiplier of the convolutional layer.
            num_estimators (int): Number of estimators in the ensemble.
            gamma (int, optional): Defaults to ``1``.
            stride (int or tuple, optional): Stride of the convolution.
                Defaults to ``1``.
            padding (int, tuple or str, optional): Padding added to all four sides
                of the input. Defaults to ``0``.
            dilation (int or tuple, optional): Spacing between kernel elements.
                Defaults to ``1``.
            groups (int, optional): Number of blocked connexions from input
                channels to output channels for each estimator. Defaults to ``1``.
            minimum_channels_per_group (int, optional): Smallest possible number of
                channels per group.
            bias (bool, optional): If ``True``, adds a learnable bias to the
                output. Defaults to ``True``.
            padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
                ``'replicate'`` or ``'circular'``. Defaults to ``'zeros'``.
            first (bool, optional): Whether this is the first layer of the
                network. Defaults to ``False``.
            last (bool, optional): Whether this is the last layer of the network.
                Defaults to ``False``.
            device (torch.device, optional): The device to use for the layer's
                parameters. Defaults to ``None``.
            dtype (torch.dtype, optional): The dtype to use for the layer's
                parameters. Defaults to ``None``.

        Explanation Note:
            Increasing :attr:`alpha` will increase the number of channels of the
            ensemble, increasing its representation capacity. Increasing
            :attr:`gamma` will increase the number of groups in the network and
            therefore reduce the number of parameters.

        Note:
            Each ensemble member will only see
            :math:`\frac{\text{in_channels}}{\text{num_estimators}}` channels,
            so when using :attr:`groups` you should make sure that
            :attr:`in_channels` and :attr:`out_channels` are both divisible by
            :attr:`num_estimators` :math:`\times`:attr:`gamma` :math:`\times`
            :attr:`groups`. However, the number of input and output channels will
            be changed to comply with this constraint.
        """
        check_packed_parameters_consistency(alpha, gamma, num_estimators)
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.num_estimators = num_estimators

        # Define the number of channels of the underlying convolution
        extended_in_channels = int(in_channels * (1 if first else alpha))
        extended_out_channels = int(
            out_channels * (num_estimators if last else alpha)
        )

        # Define the number of groups of the underlying convolution
        actual_groups = 1 if first else gamma * groups * num_estimators

        while (
            extended_in_channels % actual_groups != 0
            or extended_in_channels // actual_groups
            < minimum_channels_per_group
        ) and actual_groups // (groups * num_estimators) > 1:
            gamma -= 1
            actual_groups = gamma * groups * num_estimators

        # fix if not divisible by groups
        if extended_in_channels % actual_groups:
            extended_in_channels += (
                num_estimators - extended_in_channels % actual_groups
            )
        if extended_out_channels % actual_groups:
            extended_out_channels += (
                num_estimators - extended_out_channels % actual_groups
            )

        self.conv = nn.Conv2d(
            in_channels=extended_in_channels,
            out_channels=extended_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=actual_groups,
            bias=bias,
            padding_mode=padding_mode,
            **factory_kwargs,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

    @property
    def weight(self) -> Tensor:
        r"""The weight of the underlying convolutional layer."""
        return self.conv.weight

    @property
    def bias(self) -> Tensor | None:
        r"""The bias of the underlying convolutional layer."""
        return self.conv.bias


class PackedConv3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        alpha: int,
        num_estimators: int,
        gamma: int = 1,
        stride: _size_3_t = 1,
        padding: str | _size_3_t = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        minimum_channels_per_group: int = 64,
        bias: bool = True,
        padding_mode: str = "zeros",
        first: bool = False,
        last: bool = False,
        device: Any | None = None,
        dtype: Any | None = None,
    ) -> None:
        r"""Packed-Ensembles-style Conv3d layer.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            alpha (int): The channel multiplier of the convolutional layer.
            num_estimators (int): Number of estimators in the ensemble.
            gamma (int, optional): Defaults to ``1``.
            stride (int or tuple, optional): Stride of the convolution.
                Defaults to ``1``.
            padding (int, tuple or str, optional): Padding added to all six sides
                of the input. Defaults to ``0``.
            dilation (int or tuple, optional): Spacing between kernel elements.
                Defaults to ``1``.
            groups (int, optional): Number of blocked connexions from input
                channels to output channels for each estimator. Defaults to ``1``.
            minimum_channels_per_group (int, optional): Smallest possible number of
                channels per group.
            bias (bool, optional): If ``True``, adds a learnable bias to the
                output. Defaults to ``True``.
            padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
                ``'replicate'`` or ``'circular'``. Defaults to ``'zeros'``.
            first (bool, optional): Whether this is the first layer of the
                network. Defaults to ``False``.
            last (bool, optional): Whether this is the last layer of the network.
                Defaults to ``False``.
            device (torch.device, optional): The device to use for the layer's
                parameters. Defaults to ``None``.
            dtype (torch.dtype, optional): The dtype to use for the layer's
                parameters. Defaults to ``None``.

        Explanation Note:
            Increasing :attr:`alpha` will increase the number of channels of the
            ensemble, increasing its representation capacity. Increasing
            :attr:`gamma` will increase the number of groups in the network and
            therefore reduce the number of parameters.

        Note:
            Each ensemble member will only see
            :math:`\frac{\text{in_channels}}{\text{num_estimators}}` channels,
            so when using :attr:`groups` you should make sure that
            :attr:`in_channels` and :attr:`out_channels` are both divisible by
            :attr:`num_estimators` :math:`\times`:attr:`gamma` :math:`\times`
            :attr:`groups`. However, the number of input and output channels will
            be changed to comply with this constraint.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        check_packed_parameters_consistency(alpha, gamma, num_estimators)

        self.num_estimators = num_estimators

        # Define the number of channels of the underlying convolution
        extended_in_channels = int(in_channels * (1 if first else alpha))
        extended_out_channels = int(
            out_channels * (num_estimators if last else alpha)
        )

        # Define the number of groups of the underlying convolution
        actual_groups = 1 if first else gamma * groups * num_estimators

        while (
            extended_in_channels % actual_groups != 0
            or extended_in_channels // actual_groups
            < minimum_channels_per_group
        ) and actual_groups // (groups * num_estimators) > 1:
            gamma -= 1
            actual_groups = gamma * groups * num_estimators

        # fix if not divisible by groups
        if extended_in_channels % actual_groups:
            extended_in_channels += (
                num_estimators - extended_in_channels % actual_groups
            )
        if extended_out_channels % actual_groups:
            extended_out_channels += (
                num_estimators - extended_out_channels % actual_groups
            )

        self.conv = nn.Conv3d(
            in_channels=extended_in_channels,
            out_channels=extended_out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=actual_groups,
            bias=bias,
            padding_mode=padding_mode,
            **factory_kwargs,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)

    @property
    def weight(self) -> Tensor:
        r"""The weight of the underlying convolutional layer."""
        return self.conv.weight

    @property
    def bias(self) -> Tensor | None:
        r"""The bias of the underlying convolutional layer."""
        return self.conv.bias
