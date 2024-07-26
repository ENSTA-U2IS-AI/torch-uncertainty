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
        r"""Layer normalization over the channel dimension.

        Args:
        normalized_shape (int or list or torch.Size): input shape from an expected input
            of size

            .. math::
                [* \times \text{normalized\_shape}[0] \times \text{normalized\_shape}[1]
                    \times \ldots \times \text{normalized\_shape}[-1]]

            If a single integer is used, it is treated as a singleton list, and this module will
            normalize over the channel dimension which is expected to be of that specific size.
        eps (float): a value added to the denominator for numerical stability. Default: 1e-5
        elementwise_affine (bool): a boolean value that when set to ``True``, this module
            has learnable per-element affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.
        bias (bool): If set to ``False``, the layer will not learn an additive bias (only relevant if
            :attr:`elementwise_affine` is ``True``). Default: ``True``.
        device (torch.device or str or None): the desired device of the module.
        dtype (torch.dtype or str or None): the desired floating point type of the module.

        Attributes:
            weight: the learnable weights of the module of shape
                :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
                The values are initialized to 1.
            bias:   the learnable bias of the module of shape
                    :math:`\text{normalized\_shape}` when :attr:`elementwise_affine` is set to ``True``.
                    The values are initialized to 0.

        Shape:
            - Input: :math:`(N, *)`
            - Output: :math:`(N, *)` (same shape as input)

        """
        super().__init__(
            normalized_shape, eps, elementwise_affine, bias, device, dtype
        )
        self.cback = ChannelBack()
        self.cfront = ChannelFront()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.cfront(super().forward(self.cback(inputs)))
