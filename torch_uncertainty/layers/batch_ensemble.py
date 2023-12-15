import math

import torch
from torch import nn
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair


class BatchLinear(nn.Module):
    r"""Applies a linear transformation using BatchEnsemble method to the
    incoming data: :math:`y=(x\circ \hat{r_group})W^{T}\circ \hat{s_group} + \hat{b}`.

    Args:
        in_features (int): size of each input sample.
        out_features (int): size of each output sample.
        num_estimators (int): number of estimators in the ensemble referred as
            :math:`M` here.
        bias (bool, optional): if ``True``, adds a learnable bias to the
            output. Defaults to ``True``.

    Reference:
        Introduced by the paper `BatchEnsemble: An Alternative Approach to
        Efficient Ensemble and Lifelong Learning
        <https://arxiv.org/abs/2002.06715>`_, we present here an implementation
        of a Linear BatchEnsemble layer in `PyTorch <https://pytorch.org>`_
        heavily inspired by its `official implementation
        <https://github.com/google/edward2>`_ in `TensorFlow
        <https://www.tensorflow.org>`_.

    Attributes:
        weight:   the learnable weights (:math:`W`) of shape
            :math:`(H_{out}, H_{in})` shared between the estimators. The values
            are initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`,
            where :math:`k = \frac{1}{H_{in}}`.
        r_group: 	the learnable matrice of shape :math:`(M, H_{in})` where each row
            consist of the vector :math:`r_{i}` corresponding to the
            :math:`i^{th}` ensemble member. The values are initialized from
            :math:`\mathcal{N}(1.0, 0.5)`.
        s_group: 	the learnable matrice of shape :math:`(M, H_{out})` where each row
            consist of the vector :math:`s_{i}` corresponding to the
            :math:`i^{th}` ensemble member. The values are initialized from
            :math:`\mathcal{N}(1.0, 0.5)`.
        bias: 	the learnable bias (:math:`b`) of shape :math:`(M, H_{out})`
            where each row corresponds to the bias of the :math:`i^{th}`
            ensemble member. If :attr:`bias` is ``True``, the values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k = \frac{1}{H_{in}}`.

    Shape:
        - Input: :math:`(N, H_{in})` where :math:`N` is the batch size and
          :math:`H_{in} = \text{in_features}`.
        - Output: :math:`(N, H_{out})` where
          :math:`H_{out} = \text{out_features}`.

    Warning:
        Make sure that :math:`M | N` when calling :func:`forward()`.

    Examples:
        >>> # With three estimators
        >>> m = LinearBE(20, 30, 3)
        >>> input = torch.randn(8, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([8, 30])
    """

    __constants__ = ["in_features", "out_features", "num_estimators"]
    in_features: int
    out_features: int
    n_estimator: int
    weight: torch.Tensor
    r_group: torch.Tensor
    s_group: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_estimators: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_estimators = num_estimators

        self.linear = nn.Linear(
            in_features=in_features, out_features=out_features, bias=False
        )

        self.r_group = nn.Parameter(
            torch.empty((num_estimators, in_features), **factory_kwargs)
        )
        self.s_group = nn.Parameter(
            torch.empty((num_estimators, out_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty((num_estimators, out_features), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.r_group, mean=1.0, std=0.5)
        nn.init.normal_(self.s_group, mean=1.0, std=0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(
                self.linear.weight
            )
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.size(0)
        examples_per_estimator = torch.tensor(
            batch_size // self.num_estimators, device=inputs.device
        )
        extra = batch_size % self.num_estimators

        r_group = torch.repeat_interleave(
            self.r_group, examples_per_estimator, dim=0
        )
        r_group = torch.cat(
            [r_group, r_group[:extra]], dim=0
        )  # .unsqueeze(-1).unsqueeze(-1)
        s_group = torch.repeat_interleave(
            self.s_group, examples_per_estimator, dim=0
        )
        s_group = torch.cat(
            [s_group, s_group[:extra]], dim=0
        )  # .unsqueeze(-1).unsqueeze(-1)
        bias: torch.Tensor | None
        if self.bias is not None:
            bias = torch.repeat_interleave(
                self.bias,
                examples_per_estimator,
                dim=0,
            )
            bias = torch.cat([bias, bias[:extra]], dim=0)
        else:
            bias = None

        return self.linear(inputs * r_group) * s_group + (
            bias if bias is not None else 0
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={ self.in_features},"
            f" out_features={self.out_features},"
            f" num_estimators={self.num_estimators},"
            f" bias={self.bias is not None}"
        )


class BatchConv2d(nn.Module):
    r"""Applies a 2d convolution over an input signal composed of several input
    planes using BatchEnsemble method to the incoming data.

    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, H_{in}, W_{in})` and output
    :math:`(N, C_{out}, H_{out}, W_{out})` can be precisely described as:

    .. math::
        \text{out}(N_i, C_{\text{out}_j})=\
        &\hat{b}(N_i,C_{\text{out}_j})
        +\hat{s_group}(N_{i},C_{\text{out}_j}) \\
        &\times \sum_{k = 0}^{C_{\text{in}} - 1}
        \text{weight}(C_{\text{out}_j}, k)\star (\text{input}(N_i, k)
        \times \hat{r_group}(N_i, k))

    Reference:
        Introduced by the paper `BatchEnsemble: An Alternative Approach to
        Efficient Ensemble and Lifelong Learning
        <https://arxiv.org/abs/2002.06715>`_, we present here an implementation
        of a Conv2d BatchEnsemble layer in `PyTorch <https://pytorch.org>`_
        heavily inspired by its `official implementation
        <https://github.com/google/edward2>`_ in `TensorFlow
        <https://www.tensorflow.org>`_.

    Args:
        in_channels (int): number of channels in the input image.
        out_channels (int): number of channels produced by the convolution.
        kernel_size (int or tuple): size of the convolving kernel.
        num_estimators (int): number of estimators in the ensemble referred as
            :math:`M` here.
        stride (int or tuple, optional): stride of the convolution. Defaults to
            ``1``.
        padding (int, tuple or str, optional): padding added to all four sides
            of the input. Defaults to ``0``.
        dilation (int or tuple, optional): spacing between kernel elements.
            Defaults to ``1``.
        groups (int, optional): number of blocked connections from input
            channels to output channels. Defaults to ``1``.
        bias (bool, optional): if ``True``, adds a learnable bias to the
            output. Defaults to ``True``.

    Attributes:
        weight:     the learnable weights of the module of shape
            :math:`(\text{out_channels}, \frac{\text{in_channels}}
            {\text{groups}},`:math:`\text{kernel_size[0]},
            \text{kernel_size[1]})` shared between the estimators. The values
            of these weights are sampled from :math:`\mathcal{U}(-\sqrt{k},
            \sqrt{k})` where :math:`k = \frac{\text{groups}}{C_\text{in} *
            \prod_{i=0}^{1}\text{kernel_size}[i]}`.
        r_group: 	the learnable matrice of shape :math:`(M, C_{in})` where each row
            consist of the vector :math:`r_{i}` corresponding to the
            :math:`i^{th}` ensemble member. The values are initialized from
            :math:`\mathcal{N}(1.0, 0.5)`.
        s_group: 	the learnable matrice of shape :math:`(M, C_{out})` where each row
            consist of the vector :math:`s_{i}` corresponding to the
            :math:`i^{th}` ensemble member. The values are initialized from
            :math:`\mathcal{N}(1.0, 0.5)`.
        bias: 	the learnable bias (:math:`b`) of shape :math:`(M, C_{out})`
            where each row corresponds to the bias of the :math:`i^{th}`
            ensemble member. If :attr:`bias` is ``True``, the values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
            :math:`k=\frac{\text{groups}}{C_\text{in}*\prod_{i=0}^{1}
            \text{kernel_size}[i]}`.

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`.
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`.

        .. math::
            H_{out} = \left\lfloor\frac{H_{in} + 2 \times \text{padding}[0] -
            \text{dilation}[0] \times (\text{kernel_size}[0] - 1) - 1}
            {\text{stride}[0]} + 1\right\rfloor

        .. math::
            W_{out} = \left\lfloor\frac{W_{in}  + 2 \times \text{padding}[1] -
            \text{dilation}[1] \times (\text{kernel_size}[1] - 1) - 1}
            {\text{stride}[1]} + 1\right\rfloor

    .. warning::

        Make sure that :math:`M | N` when calling :func:`forward()`.

    Examples:
        >>> # With square kernels, four estimators and equal stride
        >>> m = Conv2dBE(3, 32, 3, 4, stride=1)
        >>> input = torch.randn(8, 3, 16, 16)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([8, 32, 14, 14])
    """

    __constants__ = [
        "stride",
        "padding",
        "dilation",
        "groups",
        "in_channels",
        "out_channels",
        "kernel_size",
        "num_estimators",
    ]
    in_channels: int
    out_channels: int
    kernel_size: tuple[int, ...]
    n_estimator: int
    stride: tuple[int, ...]
    padding: str | tuple[int, ...]
    dilation: tuple[int, ...]
    groups: int
    weight: torch.Tensor
    r_group: torch.Tensor
    s_group: torch.Tensor
    bias: torch.Tensor | None

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        num_estimators: int,
        stride: _size_2_t = 1,
        padding: str | _size_2_t = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.num_estimators = num_estimators
        self.stride = _pair(stride)
        self.padding = padding if isinstance(padding, str) else _pair(padding)
        self.dilation = _pair(dilation)

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.r_group = nn.Parameter(
            torch.empty((num_estimators, in_channels), **factory_kwargs)
        )
        self.s_group = nn.Parameter(
            torch.empty((num_estimators, out_channels), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(
                torch.empty((num_estimators, out_channels), **factory_kwargs)
            )
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where
        # k = weight.size(1) * prod(*kernel_size)
        # For more details see:
        # https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.normal_(self.r_group, mean=1.0, std=0.5)
        nn.init.normal_(self.s_group, mean=1.0, std=0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.conv.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.size(0)
        examples_per_estimator = batch_size // self.num_estimators
        extra = batch_size % self.num_estimators

        r_group = (
            torch.repeat_interleave(
                self.r_group,
                torch.full(
                    [self.num_estimators],
                    examples_per_estimator,
                    device=self.r_group.device,
                ),
                dim=0,
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        r_group = torch.cat(
            [r_group, r_group[:extra]], dim=0
        )  # .unsqueeze(-1).unsqueeze(-1)
        s_group = (
            torch.repeat_interleave(
                self.s_group,
                torch.full(
                    [self.num_estimators],
                    examples_per_estimator,
                    device=self.s_group.device,
                ),
                dim=0,
            )
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        s_group = torch.cat([s_group, s_group[:extra]], dim=0)  #

        bias: torch.Tensor | None
        if self.bias is not None:
            bias = (
                torch.repeat_interleave(
                    self.bias,
                    torch.full(
                        [self.num_estimators],
                        examples_per_estimator,
                        device=self.bias.device,
                    ),
                    dim=0,
                )
                .unsqueeze(-1)
                .unsqueeze(-1)
            )

            bias = torch.cat([bias, bias[:extra]], dim=0)
        else:
            bias = None

        return self.conv(inputs * r_group) * s_group + (
            bias if bias is not None else 0
        )

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", num_estimators={num_estimators}, stride={stride}"
        )
        return s.format(**self.__dict__)
