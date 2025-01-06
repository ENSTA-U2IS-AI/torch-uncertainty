import math
from collections.abc import Callable
from typing import Any

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t

from .functional.packed import packed_linear, packed_multi_head_attention_forward


def check_packed_parameters_consistency(alpha: float, gamma: int, num_estimators: int) -> None:
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
        raise TypeError(f"Attribute `gamma` should be an int, not {type(gamma)}")
    if gamma <= 0:
        raise ValueError(f"Attribute `gamma` should be >= 1, not {gamma}")

    if num_estimators is None:
        raise ValueError("You must specify the value of the arg. `num_estimators`")
    if not isinstance(num_estimators, int):
        raise TypeError(
            "Attribute `num_estimators` should be an int, not " f"{type(num_estimators)}"
        )
    if num_estimators <= 0:
        raise ValueError("Attribute `num_estimators` should be >= 1, not " f"{num_estimators}")


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
        extended_out_features = int(out_features * (num_estimators if last else alpha))

        # Define the number of groups of the underlying convolution
        actual_groups = num_estimators * gamma if not first else 1

        # fix if not divisible by groups
        if extended_in_features % actual_groups:
            extended_in_features += num_estimators - extended_in_features % (actual_groups)
        if extended_out_features % actual_groups:
            extended_out_features += num_estimators - extended_out_features % (actual_groups)

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
            self.bias = nn.Parameter(torch.empty(extended_out_features, **factory_kwargs))
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
            self.weight = nn.Parameter(torch.block_diag(*self.weight).to_sparse())

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
            return F.conv1d(inputs, self.weight, self.bias, 1, 0, 1, self.groups)
        return packed_linear(
            inputs=inputs,
            weight=self.weight,
            num_groups=self.groups,
            implementation=self.implementation,
            bias=self.bias,
        )


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
        extended_out_channels = int(out_channels * (num_estimators if last else alpha))

        # Define the number of groups of the underlying convolution
        actual_groups = 1 if first else gamma * groups * num_estimators

        while (
            extended_in_channels % actual_groups != 0
            or extended_in_channels // actual_groups < minimum_channels_per_group
        ) and actual_groups // (groups * num_estimators) > 1:
            gamma -= 1
            actual_groups = gamma * groups * num_estimators

        # fix if not divisible by groups
        if extended_in_channels % actual_groups:
            extended_in_channels += num_estimators - extended_in_channels % actual_groups
        if extended_out_channels % actual_groups:
            extended_out_channels += num_estimators - extended_out_channels % actual_groups

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
        extended_out_channels = int(out_channels * (num_estimators if last else alpha))

        # Define the number of groups of the underlying convolution
        actual_groups = 1 if first else gamma * groups * num_estimators

        while (
            extended_in_channels % actual_groups != 0
            or extended_in_channels // actual_groups < minimum_channels_per_group
        ) and actual_groups // (groups * num_estimators) > 1:
            gamma -= 1
            actual_groups = gamma * groups * num_estimators

        # fix if not divisible by groups
        if extended_in_channels % actual_groups:
            extended_in_channels += num_estimators - extended_in_channels % actual_groups
        if extended_out_channels % actual_groups:
            extended_out_channels += num_estimators - extended_out_channels % actual_groups

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
        extended_out_channels = int(out_channels * (num_estimators if last else alpha))

        # Define the number of groups of the underlying convolution
        actual_groups = 1 if first else gamma * groups * num_estimators

        while (
            extended_in_channels % actual_groups != 0
            or extended_in_channels // actual_groups < minimum_channels_per_group
        ) and actual_groups // (groups * num_estimators) > 1:
            gamma -= 1
            actual_groups = gamma * groups * num_estimators

        # fix if not divisible by groups
        if extended_in_channels % actual_groups:
            extended_in_channels += num_estimators - extended_in_channels % actual_groups
        if extended_out_channels % actual_groups:
            extended_out_channels += num_estimators - extended_out_channels % actual_groups

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


class PackedLayerNorm(nn.GroupNorm):
    def __init__(
        self,
        embed_dim: int,
        num_estimators: int,
        alpha: float,
        eps: float = 1e-5,
        affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        r"""Packed-Ensembles-style LayerNorm layer.

        Args:
            embed_dim (int): the number of features in the input tensor.
            num_estimators (int): the number of estimators in the ensemble.
            alpha (float): the width multiplier of the layer.
            eps (float, optional): a value added to the denominator for numerical stability. Defaults
                to 1e-5.
            affine (bool, optional): a boolean value that when set to ``True``, this module has
                learnable per_channel affine parameters initialized to ones (for weights) and zeros
                (for biases). Defaults to ``True``.
            device (torch.device, optional): the device to use for the layer's parameters. Defaults
                to ``None``.
            dtype (torch.dtype, optional): the dtype to use for the layer's parameters. Defaults to
                ``None``.

        Shape:
            - Input: :math:`(B, *)` where :math:`*` means any number of additional dimensions.
            - Output: :math:`(B, *)` (same shape as input)
        """
        super().__init__(
            num_groups=num_estimators,
            num_channels=int(embed_dim * alpha),
            eps=eps,
            affine=affine,
            device=device,
            dtype=dtype,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        x = rearrange(inputs, "b ... h -> b h ...")
        x = F.group_norm(
            x,
            self.num_groups,
            self.weight,
            self.bias,
            self.eps,
        )
        return rearrange(x, "b h ... -> b ... h")


class PackedMultiheadAttention(nn.Module):
    __constants__ = ["batch_first"]
    bias_k: Tensor | None
    bias_v: Tensor | None

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        alpha: float,
        num_estimators: int,
        gamma: int = 1,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        kdim: int | None = None,
        vdim: int | None = None,
        batch_first=False,
        first=False,
        last=False,
        device=None,
        dtype=None,
    ) -> None:
        r"""Packed-Ensembles-style MultiheadAttention layer.

        Args:
            embed_dim (int): Size of the embedding dimension.
            num_heads (int): Number of parallel attention heads.
            alpha (float): The width multiplier of the embedding dimension.
            num_estimators (int): The number of estimators packed in the layer.
            gamma (int, optional): Defaults to ``1``.
            dropout (float, optional): Dropout probability on ``attn_output_weights``. Defaults to ``0.0``
                (no dropout).
            bias (bool, optional): Ì specified, adds bias to input / output projection layers.
                Defaults to ``True``.
            add_bias_kv (bool, optional): If specified, adds bias to the key and value sequences at
                ``dim=0``. Defaults to ``False``.
            add_zero_attn (bool, optional): If specified, adds a new batch of zeros to the key and
                value sequences at ``dim=1``. Defaults to ``False``.
            kdim (int | None, optional): Total number of features for keys. Defaults to ``None``
                (uses ``kdim=embed_dim``).
            vdim (int | None, optional): Total number of features for values. Defaults to ``None``
                (uses ``vdim=embed_dim``).
            batch_first (bool, optional): If ``True``, then the input and output tensors are provided
                as (batch, seq, feature). Defaults to ``False`` (seq, batch, feature).
            first (bool, optional): Whether this is the first layer of the network. Defaults to
                ``False``.
            last (bool, optional): Whether this is the last layer of the network. Defaults to
                ``False``.
            device (torch.device, optional): The device to use for the layer's parameters. Defaults
                to ``None``.
            dtype (torch.dtype, optional): The dtype to use for the layer's parameters. Defaults to
                ``None``.

        Reference:
            - `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_: Original Multihead Attention formulation.
            - `Hierarchical Light Tranformer Ensembles for Multimodal Trajectory Forecasting <https://arxiv.org/abs/2403.17678>`_
              : Packed-Ensembles-style Multihead Attention formulation.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.embed_dim = int(embed_dim * alpha)

        augmentation = 1 if first else alpha
        in_embed_dim = int(embed_dim * augmentation)
        self.kdim = int(self.kdim * augmentation)
        self.vdim = int(self.vdim * augmentation)

        self.num_groups = 1 if first else num_estimators * gamma

        self.num_heads = num_heads * self.num_groups
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = self.embed_dim // self.num_heads
        assert (
            self.head_dim * self.num_heads == self.embed_dim
        ), "embed_dim must be divisible by num_heads"

        self.num_estimators = num_estimators
        self.alpha = alpha
        self.gamma = gamma

        if not self._qkv_same_embed_dim:
            self.q_proj_weight = nn.Parameter(
                torch.empty(
                    (
                        self.num_groups,
                        self.embed_dim // self.num_groups,
                        in_embed_dim // self.num_groups,
                    ),
                    **factory_kwargs,
                )
            )
            self.k_proj_weight = nn.Parameter(
                torch.empty(
                    (
                        self.num_groups,
                        self.embed_dim // self.num_groups,
                        self.kdim // self.num_groups,
                    ),
                    **factory_kwargs,
                )
            )
            self.v_proj_weight = nn.Parameter(
                torch.empty(
                    (
                        self.num_groups,
                        self.embed_dim // self.num_groups,
                        self.vdim // self.num_groups,
                    ),
                    **factory_kwargs,
                )
            )
            self.register_parameter("in_proj_weight", None)
        else:
            self.in_proj_weight = nn.Parameter(
                torch.empty(
                    (
                        self.num_groups,
                        3 * self.embed_dim // self.num_groups,
                        in_embed_dim // self.num_groups,
                    ),
                    **factory_kwargs,
                )
            )
            self.register_parameter("q_proj_weight", None)
            self.register_parameter("k_proj_weight", None)
            self.register_parameter("v_proj_weight", None)

        if bias:
            self.in_proj_bias = nn.Parameter(torch.empty(3 * self.embed_dim, **factory_kwargs))
        else:
            self.register_parameter("in_proj_bias", None)

        if add_bias_kv:
            self.bias_k = nn.Parameter(torch.empty((1, 1, self.embed_dim), **factory_kwargs))
            self.bias_v = nn.Parameter(torch.empty((1, 1, self.embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.out_proj = PackedLinear(
            in_features=embed_dim,
            out_features=embed_dim,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            implementation="einsum",
            bias=bias,
            first=False,
            last=last,
            **factory_kwargs,
        )

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            for i in range(self.in_proj_weight.size(0)):
                nn.init.xavier_uniform_(self.in_proj_weight[i])
        else:
            for i in range(self.q_proj_weight.size(0)):
                nn.init.xavier_uniform_(self.q_proj_weight[i])
                nn.init.xavier_uniform_(self.k_proj_weight[i])
                nn.init.xavier_uniform_(self.v_proj_weight[i])

        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        key_padding_mask: Tensor | None = None,
        need_weights: bool = False,
        attn_mask: Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[Tensor, None]:
        r"""Computes attention outputs given query, key, and value tensors.

        Args:
            query (Tensor): Query embeddings of shape :math:`(L, E_q)` for unbatched input,
                :math:`(L, B, E_q)` when ``batch_first=False`` or :math:`(B, L, E_q)` when
                ``batch_first=True``, where :math:`L` is the target sequence length, :math:`B` is
                the batch size, and :math:`E_q` is the query embedding dimension ``embed_dim``.
            key (Tensor): Key embeddingd of shape :math:`(S, E_k)` for unbatched input,
                :math:`(S, B, E_k)` when ``batch_first=False`` or :math:`(B, S, E_k)` when
                ``batch_first=True``, where :math:`S` is the source sequence length, :math:`B` is
                the batch size and :math:`E_k` is the key embedding dimension ``kdim``.
            value (Tensor): Value embeddings of shape :math:`(S, E_v)` for unbatched input,
                :math:`(S, B, E_v)` when ``batch_first=False`` or :math:`(B, S, E_v)` when
                ``batch_first=True``, where :math:`S` is the source sequence length, :math:`B` is
                the batch size and :math:`E_v` is the value embedding dimension ``vdim``.
            key_padding_mask (Tensor | None, optional): If specified, a mask of shape
                :math:`(B, S)` indicating which elements within ``key`` to ignore for the purpose
                of attention (i.e. treat as "padding"). For unbatched `query`, shape should be
                :math:`(S)`. Binary and float masks are supported. For a binary mask, a ``True``
                value indicates that the corresponding ``key`` value will be ignored for the
                purpose of attention. For a float mask, it will be directly added to the
                corresponding ``key`` value. Defaults to ``None``.
            need_weights (bool, optional): If specified, returns ``attn_output_weights`` in
                addition to ``attn_outputs``. Set ``need_weights=False`` to use the optimized
                ``scale_dot_product_attention`` and achieve the best performance for MHA.
                Defaults to ``False``.
            attn_mask (Tensor | None, optional): If specified, a 2D or 3D mask preventing attention
                to certain positions. Must be of shape :math:`(L,S)` or
                :math:`(B \times \text{num_heads}, L, S)`, where :math:`B` is the batch size, :math:`L`
                is the target sequence length, and :math:`S` is the source sequence length. A 2D mask
                will be broadcasted across the batch while a 3D mask allows for a different mask for
                each entry in the batch. Binary and float masks are supported. For a binary mask, a
                ``True`` value indicates that the corresponding position is not allowed to attend to.
                For a float mask, the mask values will be added to the attention weight. If both
                ``attn_mask`` and ``key_padding_mask`` are provided, their types should match.
                Defaults to ``None``.
            average_attn_weights (bool, optional): If ``True``, indicates that the returned
                ``attn_weights`` should be averaged across heads. Otherwise, ``attn_weights`` are
                provided separately per head. Note that this flag only has an effect when
                ``need_weights=True``. Defaults to ``True``.
            is_causal (bool, optional): _description_. Defaults to ``False``.

        Warning:
            ``need_weights=True`` and therefore ``average_attn_weights`` are not supported yet thus
            have no effect.

        Returns:
            tuple[Tensor, None]:
                - *attn_output* (Tensor): The output tensor of shape :math:`(L, E_q)`, :math:`(L, B, E_q)`
                  or :math:`(B, L, E_q)` where :math:`L` is the target sequence length, :math:`B` is
                  the batch size, and :math:`E_q` is the embedding dimension ``embed_dim``.
                - *attn_output_weights* (None): Always ``None`` has we do not support
                  ``need_weights=True`` yet.
        """
        is_batched = query.dim() == 3

        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype,
        )

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=None,
            other_name="",
            target_type=query.dtype,
            check_other=False,
        )

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = (x.transpose(1, 0) for x in (query, key))
                    value = key
            else:
                query, key, value = (x.transpose(1, 0) for x in (query, key, value))

        if not self._qkv_same_embed_dim:
            (
                attn_output,
                _,
            ) = packed_multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.num_groups,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        else:
            (
                attn_output,
                _,
            ) = packed_multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.num_groups,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal,
            )
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
        return attn_output, None


class PackedTransformerEncoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        alpha: float,
        num_estimators: int,
        gamma: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        batch_first: bool = False,
        norm_first: bool = False,
        first: bool = False,
        last: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        r"""Packed-Ensembles-style TransformerEncoderLayer (made up of self-attention followed by a
        feedforward network).

        Args:
            d_model (int): the number of expected features in the input.
            nhead (int): the number of heads in the multiheadattention models.
            alpha (float): the width multiplier of the layer.
            num_estimators (int): the number of estimators packed in the layer.
            gamma (int, optional): Defaults to ``1``.
            dim_feedforward (int, optional): the dimension of the feedforward network model. Defaults
                to ``2048``.
            dropout (float, optional): the dropout value. Defaults to ``0.1``.
            activation (Callable[[Tensor], Tensor], optional): the activation function of the
                intermediate layer, that is a unary callable. Defaults to ``F.relu``.
            layer_norm_eps (float, optional): the eps value in layer normalization components. Defaults
                to ``1e-5``.
            bias (bool, optional): If ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an
                additive bias. Defaults to ``True``.
            batch_first (bool, optional): If ``True``, then the input and output tensors are provided
                as :math:`(\text{batch}, \text{seq}, \text{d_model})`. Defaults to ``False``
                :math:`(\text{seq}, \text{batch}, \text{d_model})`.
            norm_first (bool, optional): If ``True``, the layer norm is done prior to attention and
                feedforward operations, respectively. Otherwise, it is done after. Defaults to
                ``False``.
            first (bool, optional): Whether this is the first layer of the network. Defaults to
                ``False``.
            last (bool, optional): Whether this is the last layer of the network. Defaults to
                ``False``.
            device (torch.device, optional): The device to use for the layer's parameters. Defaults
                to ``None``.
            dtype (torch.dtype, optional): The dtype to use for the layer's parameters. Defaults to
                ``None``.

        Reference:
            - `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_: Original Multihead Attention formulation.
            - `Hierarchical Light Tranformer Ensembles for Multimodal Trajectory Forecasting <https://arxiv.org/abs/2403.17678>`_
              : Packed-Ensembles-style Multihead Attention formulation.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.self_attn = PackedMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            alpha=alpha,
            num_estimators=num_estimators,
            bias=bias,
            gamma=gamma,
            dropout=dropout,
            batch_first=batch_first,
            first=first,
            **factory_kwargs,
        )

        self.linear1 = PackedLinear(
            in_features=d_model,
            out_features=dim_feedforward,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            implementation="einsum",
            bias=bias,
            **factory_kwargs,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = PackedLinear(
            in_features=dim_feedforward,
            out_features=d_model,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            implementation="einsum",
            last=last,
            bias=bias,
            **factory_kwargs,
        )

        self.norm_first = norm_first
        if self.norm_first and first:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        else:
            self.norm1 = PackedLayerNorm(
                embed_dim=d_model,
                num_estimators=num_estimators,
                alpha=alpha,
                eps=layer_norm_eps,
                **factory_kwargs,
            )

        if not self.norm_first and last:
            self.norm2 = PackedLayerNorm(
                embed_dim=d_model,
                num_estimators=num_estimators,
                alpha=alpha,
                eps=layer_norm_eps,
                **factory_kwargs,
            )
        else:
            self.norm2 = PackedLayerNorm(
                embed_dim=d_model,
                num_estimators=num_estimators,
                alpha=alpha,
                eps=layer_norm_eps,
                **factory_kwargs,
            )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def forward(
        self,
        src: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src (Tensor): The sequence to the encoder layer. Shape: :math:`(B, L, E)` or
                :math:`(L, B, E)`.
            src_mask (Tensor | None, optional): The mask for the ``src`` sequence. Defaults to ``None``.
            src_key_padding_mask (Tensor | None, optional): The mask for the ``src`` keys per
                batch. Defaults to ``None``.
            is_causal (bool, optional): If specified, applies a causal mask as ``src_mask``.
                Defaults to ``False``. Warning: ``is_causal`` provides a hint the ``src_mask`` is
                a causal mask. Providing incorrect hints can result in incorrect execution,
                including forward and backward compatibility.

        Returns:
            Tensor: The output of the encoder layer. Shape: :math:`(B, L, E)` or :math:`(L, B, E)`.
        """
        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype,
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        x = src
        if self.norm_first:
            x = x + self._sa_block(
                self.norm1(x),
                src_mask,
                src_key_padding_mask,
                is_causal=is_causal,
            )
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(
                x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal=is_causal)
            )
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    # feed-forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)


class PackedTransformerDecoderLayer(nn.Module):
    __constants__ = ["batch_first", "norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        alpha: int,
        num_estimators: int,
        gamma: int = 1,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: Callable[[Tensor], Tensor] = F.relu,
        layer_norm_eps: float = 1e-5,
        batch_first: bool = False,
        norm_first: bool = False,
        first: bool = False,
        last: bool = False,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        r"""Packed-Ensembles-style TransformerDecoderLayer (made up of self-attention, multi-head
        attention, and feedforward network).

        Args:
            d_model (int): the number of expected features in the input.
            nhead (int): the number of heads in the multiheadattention models.
            alpha (float): the width multiplier of the layer.
            num_estimators (int): the number of estimators packed in the layer.
            gamma (int, optional): Defaults to ``1``.
            dim_feedforward (int, optional): the dimension of the feedforward network model. Defaults
                to ``2048``.
            dropout (float, optional): the dropout value. Defaults to ``0.1``.
            activation (Callable[[Tensor], Tensor], optional): the activation function of the
                intermediate layer, that is a unary callable. Defaults to ``F.relu``.
            layer_norm_eps (float, optional): the eps value in layer normalization components. Defaults
                to ``1e-5``.
            bias (bool, optional): If ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an
                additive bias. Defaults to ``True``.
            batch_first (bool, optional): If ``True``, then the input and output tensors are provided
                as :math:`(\text{batch}, \text{seq}, \text{d_model})`. Defaults to ``False``
                :math:`(\text{seq}, \text{batch}, \text{d_model})`.
            norm_first (bool, optional): If ``True``, the layer norm is done prior to attention and
                feedforward operations, respectively. Otherwise, it is done after. Defaults to
                ``False``.
            first (bool, optional): Whether this is the first layer of the network. Defaults to
                ``False``.
            last (bool, optional): Whether this is the last layer of the network. Defaults to
                ``False``.
            device (torch.device, optional): The device to use for the layer's parameters. Defaults
                to ``None``.
            dtype (torch.dtype, optional): The dtype to use for the layer's parameters. Defaults to
                ``None``.

        Reference:
            - `Attention Is All You Need <https://arxiv.org/abs/1706.03762>`_: Original Multihead Attention formulation.
            - `Hierarchical Light Tranformer Ensembles for Multimodal Trajectory Forecasting <https://arxiv.org/abs/2403.17678>`_
              : Packed-Ensembles-style Multihead Attention formulation.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.self_attn = PackedMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            first=first,
            **factory_kwargs,
        )

        self.multihead_attn = PackedMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            dropout=dropout,
            bias=bias,
            batch_first=batch_first,
            **factory_kwargs,
        )

        self.linear1 = PackedLinear(
            in_features=d_model,
            out_features=dim_feedforward,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            implementation="einsum",
            bias=bias,
            **factory_kwargs,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear2 = PackedLinear(
            in_features=dim_feedforward,
            out_features=d_model,
            alpha=alpha,
            num_estimators=num_estimators,
            gamma=gamma,
            implementation="einsum",
            bias=bias,
            last=last,
            **factory_kwargs,
        )

        self.norm_first = norm_first
        if self.norm_first and first:
            self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps)
        else:
            self.norm1 = PackedLayerNorm(
                embed_dim=d_model,
                num_estimators=num_estimators,
                alpha=alpha,
                eps=layer_norm_eps,
                **factory_kwargs,
            )

        self.norm2 = PackedLayerNorm(
            embed_dim=d_model,
            num_estimators=num_estimators,
            alpha=alpha,
            eps=layer_norm_eps,
            **factory_kwargs,
        )

        if not self.norm_first and last:
            self.norm3 = PackedLayerNorm(
                embed_dim=d_model,
                num_estimators=num_estimators,
                alpha=num_estimators,
                eps=layer_norm_eps,
                **factory_kwargs,
            )
        else:
            self.norm3 = PackedLayerNorm(
                embed_dim=d_model,
                num_estimators=num_estimators,
                alpha=alpha,
                eps=layer_norm_eps,
                **factory_kwargs,
            )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        if activation is F.relu or isinstance(activation, torch.nn.ReLU):
            self.activation_relu_or_gelu = 1
        elif activation is F.gelu or isinstance(activation, torch.nn.GELU):
            self.activation_relu_or_gelu = 2
        else:
            self.activation_relu_or_gelu = 0
        self.activation = activation

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Tensor | None = None,
        memory_mask: Tensor | None = None,
        tgt_key_padding_mask: Tensor | None = None,
        memory_key_padding_mask: Tensor | None = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the input (and mask) through the decoder layer.

        Args:
            tgt (Tensor): The sequence to the decoder layer. Shape: :math:`(B, L, E)` or
                :math:`(L, B, E)`.
            memory (Tensor): The sequence from the last layer of the encoder. Shape:
                :math:`(B, S, E)` or :math:`(S, B, E)`.
            tgt_mask (Tensor | None, optional): The mask for the ``tgt`` sequence. Defaults to
                ``None``.
            memory_mask (Tensor | None, optional): The mask for the ``memory`` sequence. Defaults
                to ``None``.
            tgt_key_padding_mask (Tensor | None, optional): The mask for the ``tgt`` keys per
                batch. Defaults to ``None``.
            memory_key_padding_mask (Tensor | None, optional): The mask for the ``memory`` keys per
                batch. Defaults to ``None``.
            tgt_is_causal (bool, optional): If specified, applies a causal mask as ``tgt_mask``.
                Defaults to ``False``. Warning: ``tgt_is_causal`` provides a hint the ``tgt_mask``
                is a causal mask. Providing incorrect hints can result in incorrect execution,
                including forward and backward compatibility.
            memory_is_causal (bool, optional): If specified, applies a causal mask as ``memory_mask``.
                Defaults to ``False``. Warning: ``memory_is_causal`` provides a hint the ``memory_mask``
                is a causal mask. Providing incorrect hints can result in incorrect execution,
                including forward and backward compatibility.

        Returns:
            Tensor: The output of the encoder layer. Shape: :math:`(B, L, E)` or :math:`(L, B, E)`.
        """
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block(
                self.norm2(x),
                memory,
                memory_mask,
                memory_key_padding_mask,
                memory_is_causal,
            )
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(
                x
                + self._mha_block(
                    x,
                    memory,
                    memory_mask,
                    memory_key_padding_mask,
                    memory_is_causal,
                )
            )
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(
        self,
        x: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )[0]
        return self.dropout1(x)

    # multi-head attention block
    def _mha_block(
        self,
        x: Tensor,
        memory: Tensor,
        attn_mask: Tensor | None,
        key_padding_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        x = self.multihead_attn(
            x,
            memory,
            memory,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            is_causal=is_causal,
        )[0]
        return self.dropout2(x)

    # feed-forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
