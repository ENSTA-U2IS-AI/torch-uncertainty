import torch
from torch import Tensor
from torch.nn import LayerNorm

from .utils import ChannelBack, ChannelFront


class ChannelLayerNorm(LayerNorm):
    def __init__(
        self,
        normalized_shape: int | list[int],
        eps: float = 0.00001,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | str | None = None,
    ) -> None:
        r"""Masksembles-style Conv2d layer.

        Args:
            in_channels (int): Number of channels in the input image.
            out_channels (int): Number of channels produced by the convolution.
            kernel_size (int or tuple): Size of the convolving kernel.
            num_estimators (int): Number of estimators in the ensemble.
            scale (float): The scale parameter for the masks.
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
            normalized_shape (int | list[int]): Shape of the input to normalize.
            elementwise_affine (bool, optional): Defaults to ``True``.
            eps (float): A small constant added to the denominator for numerical stability. Default is 1e-5.
            device (Any, optional): The desired device of returned tensor.
            Defaults to ``None``.
            dtype (Any, optional): The desired data type of returned tensor.
            Defaults to ``None``.

        Warning:
            Be sure to apply a repeat on the batch at the start of the training
            if you use `MaskedConv2d`.

        Reference:
            `Masksembles for Uncertainty Estimation`, Nikita Durasov, Timur
            Bagautdinov, Pierre Baque, Pascal Fua.
        """
        super().__init__(normalized_shape, eps, elementwise_affine, bias, device, dtype)
        self.cback = ChannelBack()
        self.cfront = ChannelFront()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.cfront(super().forward(self.cback(inputs)))
