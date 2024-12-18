import inspect

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def get_dist_linear_layer(dist_family: str) -> type[nn.Module]:
    if dist_family == "normal":
        return NormalLinear
    if dist_family == "laplace":
        return LaplaceLinear
    if dist_family == "cauchy":
        return CauchyLinear
    if dist_family == "student":
        return StudentTLinear
    if dist_family == "nig":
        return NormalInverseGammaLinear
    raise NotImplementedError(
        f"{dist_family} distribution is not supported. Raise an issue if needed."
    )


def get_dist_conv_layer(dist_family: str) -> type[nn.Module]:
    if dist_family == "normal":
        return NormalConvNd
    if dist_family == "laplace":
        return LaplaceConvNd
    if dist_family == "cauchy":
        return CauchyConvNd
    if dist_family == "student":
        return StudentTConvNd
    if dist_family == "nig":
        return NormalInverseGammaConvNd
    raise NotImplementedError(
        f"{dist_family} distribution is not supported. Raise an issue if needed."
    )


class _ExpandOutputLinear(nn.Module):
    """Abstract class for expanding the output of any nn.Module using an `out_features` argument.

    Args:
        base_layer (type[nn.Module]): The base layer class.
        event_dim (int): The number of event dimensions.
        num_params (int): The number of parameters to output. For instance, the normal distribution
            has 2 parameters (loc and scale).
        **layer_args: Additional arguments for the base layer.
    """

    def __init__(self, base_layer: type[nn.Module], event_dim: int, num_params: int, **layer_args):
        if "out_features" not in inspect.getfullargspec(base_layer.__init__).args:
            raise ValueError(f"{base_layer.__name__} does not have an `out_features` argument.")

        super().__init__()
        self.base_layer = base_layer(out_features=num_params * event_dim, **layer_args)
        self.event_dim = event_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.base_layer(x)


class _ExpandOutputConvNd(nn.Module):
    """Abstract class for expanding the output of any nn.Module using an `out_channels` argument.

    Args:
        base_layer (type[nn.Module]): The base layer class.
        event_dim (int): The number of event dimensions.
        num_params (int): The number of parameters to output. For instance, the normal distribution
            has 2 parameters (loc and scale).
        **layer_args: Additional arguments for the base layer.
    """

    def __init__(self, base_layer: type[nn.Module], event_dim: int, num_params: int, **layer_args):
        if "out_channels" not in inspect.getfullargspec(base_layer.__init__).args:
            raise ValueError(f"{base_layer.__name__} does not have an `out_channels` argument.")

        super().__init__()
        self.base_layer = base_layer(out_channels=num_params * event_dim, **layer_args)
        self.event_dim = event_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.base_layer(x)


class _LocScaleLinear(_ExpandOutputLinear):
    """Base Linear layer for any distribution with loc and scale parameters.

    Args:
        base_layer (type[nn.Module]): The base layer class.
        event_dim (int): The number of event dimensions.
        min_scale (float): The minimal value of the scale parameter.
        **layer_args: Additional arguments for the base layer.
    """

    def __init__(
        self,
        base_layer: type[nn.Module],
        event_dim: int,
        min_scale: float = 1e-6,
        **layer_args,
    ) -> None:
        super().__init__(
            base_layer=base_layer,
            event_dim=event_dim,
            num_params=2,
            **layer_args,
        )
        self.min_scale = min_scale

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = super().forward(x)
        loc = x[..., : self.event_dim]
        scale = torch.clamp(
            F.softplus(x[..., self.event_dim : 2 * self.event_dim]), min=self.min_scale
        )
        return {"loc": loc, "scale": scale}


class _LocScaleConvNd(_ExpandOutputConvNd):
    """Base Convolutional layer for any distribution with loc and scale parameters.

    Args:
        base_layer (type[nn.Module]): The base layer class.
        event_dim (int): The number of event dimensions.
        min_scale (float): The minimal value of the scale parameter.
        **layer_args: Additional arguments for the base layer.
    """

    def __init__(
        self,
        base_layer: type[nn.Module],
        event_dim: int,
        min_scale: float = 1e-6,
        **layer_args,
    ) -> None:
        super().__init__(
            base_layer=base_layer,
            event_dim=event_dim,
            num_params=2,
            **layer_args,
        )
        self.min_scale = min_scale

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = super().forward(x)
        loc = x[:, : self.event_dim]
        scale = torch.clamp(
            F.softplus(x[:, self.event_dim : 2 * self.event_dim]), min=self.min_scale
        )
        return {"loc": loc, "scale": scale}


class NormalLinear(_LocScaleLinear):
    r"""Normal Distribution Linear Density Layer.

    Args:
        base_layer (type[nn.Module]): The base layer class.
        event_dim (int): The number of event dimensions.
        min_scale (float): The minimal value of the scale parameter.
        **layer_args: Additional arguments for the base layer.

    Shape:
        - Input: :math:`(\ast, H_{in})` where :math:`\ast` means any number of dimensions including
          none and :math:`H_{in} = \text{in_features}`.
        - Output: A dict with the following keys

          - ``"loc"``: The mean of the Normal distribution of shape :math:`(\ast, H_{out})` where
            all but the last dimension are the same as the input and
            :math:`H_{out} = \text{out_features}`.
          - ``"scale"``: The standard deviation of the Normal distribution of shape
            :math:`(\ast, H_{out})`.
    """


class NormalConvNd(_LocScaleConvNd):
    r"""Normal Distribution Convolutional Density Layer.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of event channels.
        kernel_size (int | tuple[int]): The size of the convolutional kernel.
        stride (int | tuple[int]): The stride of the convolution.
        padding (int | tuple[int]): The padding of the convolution.
        dilation (int | tuple[int]): The dilation of the convolution.
        groups (int): The number of groups in the convolution.
        min_scale (float): The minimal value of the scale parameter.
        device (torch.device): The device where the layer is stored.
        dtype (torch.dtype): The datatype of the layer.

    Shape:
        - Input: :math:`(N, C_{in}, \ast)` where :math:`\ast` means any number of dimensions and
          :math:`C_{in} = \text{in_channels}` and :math:`N` is the batch size.
        - Output: A dict with the following keys

          - ``"loc"``: The mean of the Normal distribution of shape :math:`(N, C_{out}, \ast)` where
            :math:`C_{out} = \text{out_channels}`.
          - ``"scale"``: The standard deviation of the Normal distribution of shape
            :math:`(\ast, C_{out}, \ast)`.
    """


class LaplaceLinear(_LocScaleLinear):
    r"""Laplace Distribution Linear Density Layer.

    Args:
        base_layer (type[nn.Module]): The base layer class.
        event_dim (int): The number of event dimensions.
        min_scale (float): The minimal value of the scale parameter.
        **layer_args: Additional arguments for the base layer.

    Shape:
        - Input: :math:`(\ast, H_{in})` where :math:`\ast` means any number of dimensions including
          none and :math:`H_{in} = \text{in_features}`.
        - Output: A dict with the following keys

          - ``"loc"``: The mean of the Laplace distribution of shape :math:`(\ast, H_{out})` where
            all but the last dimension are the same as the input and
            :math:`H_{out} = \text{out_features}`.
          - ``"scale"``: The standard deviation of the Laplace distribution of shape
            :math:`(\ast, H_{out})`.
    """


class LaplaceConvNd(_LocScaleConvNd):
    r"""Laplace Distribution Convolutional Density Layer.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of event channels.
        kernel_size (int | tuple[int]): The size of the convolutional kernel.
        stride (int | tuple[int]): The stride of the convolution.
        padding (int | tuple[int]): The padding of the convolution.
        dilation (int | tuple[int]): The dilation of the convolution.
        groups (int): The number of groups in the convolution.
        min_scale (float): The minimal value of the scale parameter.
        device (torch.device): The device where the layer is stored.
        dtype (torch.dtype): The datatype of the layer.

    Shape:
        - Input: :math:`(N, C_{in}, \ast)` where :math:`\ast` means any number of dimensions and
          :math:`C_{in} = \text{in_channels}` and :math:`N` is the batch size.
        - Output: A dict with the following keys

          - ``"loc"``: The mean of the Laplace distribution of shape :math:`(N, C_{out}, \ast)` where
            :math:`C_{out} = \text{out_channels}`.
          - ``"scale"``: The standard deviation of the Laplace distribution of shape
            :math:`(\ast, C_{out}, \ast)`.
    """


class CauchyLinear(_LocScaleLinear):
    r"""Cauchy Distribution Linear Density Layer.

    Args:
        base_layer (type[nn.Module]): The base layer class.
        event_dim (int): The number of event dimensions.
        min_scale (float): The minimal value of the scale parameter.
        **layer_args: Additional arguments for the base layer.

    Shape:
        - Input: :math:`(\ast, H_{in})` where :math:`\ast` means any number of dimensions including
          none and :math:`H_{in} = \text{in_features}`.
        - Output: A dict with the following keys

          - ``"loc"``: The mean of the Cauchy distribution of shape :math:`(\ast, H_{out})` where
            all but the last dimension are the same as the input and
            :math:`H_{out} = \text{out_features}`.
          - ``"scale"``: The standard deviation of the Cauchy distribution of shape
            :math:`(\ast, H_{out})`.
    """


class CauchyConvNd(_LocScaleConvNd):
    r"""Cauchy Distribution Convolutional Density Layer.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of event channels.
        kernel_size (int | tuple[int]): The size of the convolutional kernel.
        stride (int | tuple[int]): The stride of the convolution.
        padding (int | tuple[int]): The padding of the convolution.
        dilation (int | tuple[int]): The dilation of the convolution.
        groups (int): The number of groups in the convolution.
        min_scale (float): The minimal value of the scale parameter.
        device (torch.device): The device where the layer is stored.
        dtype (torch.dtype): The datatype of the layer.

    Shape:
        - Input: :math:`(N, C_{in}, \ast)` where :math:`\ast` means any number of dimensions and
          :math:`C_{in} = \text{in_channels}` and :math:`N` is the batch size.
        - Output: A dict with the following keys

          - ``"loc"``: The mean of the Cauchy distribution of shape :math:`(N, C_{out}, \ast)` where
            :math:`C_{out} = \text{out_channels}`.
          - ``"scale"``: The standard deviation of the Cauchy distribution of shape
            :math:`(\ast, C_{out}, \ast)`.
    """


class StudentTLinear(_ExpandOutputLinear):
    r"""Student's T-Distribution Linear Density Layer.

    Args:
        base_layer (type[nn.Module]): The base layer class.
        event_dim (int): The number of event dimensions.
        min_scale (float): The minimal value of the scale parameter.
        min_df (float): The minimal value of the degrees of freedom parameter.
        fixed_df (float): If not None, the degrees of freedom parameter is fixed to this value.
            Otherwise, it is learned.
        **layer_args: Additional arguments for the base layer.

    Shape:
        - Input: :math:`(\ast, H_{in})` where :math:`\ast` means any number of dimensions including
          none and :math:`H_{in} = \text{in_features}`.
        - Output: A dict with the following keys

          - ``"loc"``: The mean of the Student's t-distribution of shape :math:`(\ast, H_{out})` where
            all but the last dimension are the same as the input and
            :math:`H_{out} = \text{out_features}`.
          - ``"scale"``: The standard deviation of the Student's t-distribution of shape
            :math:`(\ast, H_{out})`.
          - ``"df"``: The degrees of freedom of the Student's t distribution of shape
            :math:`(\ast, H_{out})` or Number.
    """

    def __init__(
        self,
        base_layer: type[nn.Module],
        event_dim: int,
        min_scale: float = 1e-6,
        min_df: float = 2.0,
        fixed_df: float | None = None,
        **layer_args,
    ) -> None:
        super().__init__(
            base_layer=base_layer,
            event_dim=event_dim,
            num_params=3 if fixed_df is None else 2,
            **layer_args,
        )

        self.min_scale = min_scale
        self.min_df = min_df
        self.fixed_df = fixed_df

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = super().forward(x)
        loc = x[..., : self.event_dim]
        scale = torch.clamp(
            F.softplus(x[..., self.event_dim : 2 * self.event_dim]), min=self.min_scale
        )
        df = (
            torch.clamp(F.softplus(x[..., 2 * self.event_dim :]), min=self.min_df)
            if self.fixed_df is None
            else torch.full_like(loc, self.fixed_df)
        )
        return {"loc": loc, "scale": scale, "df": df}


class StudentTConvNd(_ExpandOutputConvNd):
    r"""Student's T-Distribution Convolutional Density Layer.

    Args:
        base_layer (type[nn.Module]): The base layer class.
        event_dim (int): The number of event dimensions.
        min_scale (float): The minimal value of the scale parameter.
        min_df (float): The minimal value of the degrees of freedom parameter.
        fixed_df (float): If not None, the degrees of freedom parameter is fixed to this value.
            Otherwise, it is learned.
        **layer_args: Additional arguments for the base layer.

    Shape:
        - Input: :math:`(N, C_{in}, \ast)` where :math:`\ast` means any number of dimensions and
          :math:`C_{in} = \text{in_channels}` and :math:`N` is the batch size.
        - Output: A dict with the following keys

          - ``"loc"``: The mean of the Student's t-distribution of shape :math:`(N, C_{out}, \ast)` where
            :math:`C_{out} = \text{out_channels}`.
          - ``"scale"``: The standard deviation of the Student's t-distribution of shape
            :math:`(\ast, C_{out}, \ast)`.
          - ``"df"``: The degrees of freedom of the Student's t distribution of shape
            :math:`(\ast, C_{out}, \ast)`.
    """

    def __init__(
        self,
        base_layer: type[nn.Module],
        event_dim: int,
        min_scale: float = 1e-6,
        min_df: float = 2.0,
        fixed_df: float | None = None,
        **layer_args,
    ) -> None:
        super().__init__(
            base_layer=base_layer,
            event_dim=event_dim,
            num_params=3 if fixed_df is None else 2,
            **layer_args,
        )

        self.min_scale = min_scale
        self.min_df = min_df
        self.fixed_df = fixed_df

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = super().forward(x)
        loc = x[:, : self.event_dim]
        scale = torch.clamp(
            F.softplus(x[:, self.event_dim : 2 * self.event_dim]), min=self.min_scale
        )
        df = (
            torch.clamp(F.softplus(x[:, 2 * self.event_dim :]), min=self.min_df)
            if self.fixed_df is None
            else torch.full_like(loc, self.fixed_df)
        )
        return {"loc": loc, "scale": scale, "df": df}


class NormalInverseGammaLinear(_ExpandOutputLinear):
    r"""Normal-Inverse-Gamma Distribution Linear Density Layer.

    Args:
        base_layer (type[nn.Module]): The base layer class.
        event_dim (int): The number of event dimensions.
        min_lmbda (float): The minimal value of the :math:`\lambda` parameter.
        min_alpha (float): The minimal value of the :math:`\alpha` parameter.
        min_beta (float): The minimal value of the :math:`\beta` parameter.
        **layer_args: Additional arguments for the base layer.

    Shape:
        - Input: :math:`(\ast, H_{in})` where :math:`\ast` means any number of dimensions including
          none and :math:`H_{in} = \text{in_features}`.
        - Output: A dict with the following keys

          - ``"loc"``: The mean of the Normal-Inverse-Gamma distribution of shape :math:`(\ast, H_{out})` where
            all but the last dimension are the same as the input and
            :math:`H_{out} = \text{out_features}`.
          - ``"lmbda"``: The lambda parameter of the Normal-Inverse-Gamma distribution of shape
            :math:`(\ast, H_{out})`.
          - ``"alpha"``: The alpha parameter of the Normal-Inverse-Gamma distribution of shape
            :math:`(\ast, H_{out})`.
          - ``"beta"``: The beta parameter of the Normal-Inverse-Gamma distribution of shape
            :math:`(\ast, H_{out})`.

    Source:
        - `Normal-Inverse-Gamma Distribution <https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution>`_
    """

    def __init__(
        self,
        base_layer: type[nn.Module],
        event_dim: int,
        min_lmbda: float = 1e-6,
        min_alpha: float = 1e-6,
        min_beta: float = 1e-6,
        **layer_args,
    ) -> None:
        super().__init__(
            base_layer=base_layer,
            event_dim=event_dim,
            num_params=4,
            **layer_args,
        )

        self.min_lmbda = min_lmbda
        self.min_alpha = min_alpha
        self.min_beta = min_beta

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = super().forward(x)
        loc = x[..., : self.event_dim]
        lmbda = torch.clamp(
            F.softplus(x[..., self.event_dim : 2 * self.event_dim]), min=self.min_lmbda
        )
        alpha = 1 + torch.clamp(
            F.softplus(x[..., 2 * self.event_dim : 3 * self.event_dim]), min=self.min_alpha
        )
        beta = torch.clamp(F.softplus(x[..., 3 * self.event_dim :]), min=self.min_beta)
        return {
            "loc": loc,
            "lmbda": lmbda,
            "alpha": alpha,
            "beta": beta,
        }


class NormalInverseGammaConvNd(_ExpandOutputConvNd):
    r"""Normal-Inverse-Gamma Distribution Convolutional Density Layer.

    Args:
        base_layer (type[nn.Module]): The base layer class.
        event_dim (int): The number of event dimensions.
        min_lmbda (float): The minimal value of the :math:`\lambda` parameter.
        min_alpha (float): The minimal value of the :math:`\alpha` parameter.
        min_beta (float): The minimal value of the :math:`\beta` parameter.
        **layer_args: Additional arguments for the base layer.

    Shape:
        - Input: :math:`(N, C_{in}, \ast)` where :math:`\ast` means any number of dimensions and
          :math:`C_{in} = \text{in_channels}` and :math:`N` is the batch size.
        - Output: A dict with the following keys

          - ``"loc"``: The mean of the Normal-Inverse-Gamma distribution of shape :math:`(N, C_{out}, \ast)` where
            :math:`C_{out} = \text{out_channels}`.
          - ``"lmbda"``: The lambda parameter of the Normal-Inverse-Gamma distribution of shape
            :math:`(N, C_{out}, \ast)`.
          - ``"alpha"``: The alpha parameter of the Normal-Inverse-Gamma distribution of shape
            :math:`(N, C_{out}, \ast)`.
          - ``"beta"``: The beta parameter of the Normal-Inverse-Gamma distribution of shape
            :math:`(N, C_{out}, \ast)`.

    Source:
        - `Normal-Inverse-Gamma Distribution <https://en.wikipedia.org/wiki/Normal-inverse-gamma_distribution>`_
    """

    def __init__(
        self,
        base_layer: type[nn.Module],
        event_dim: int,
        min_lmbda: float = 1e-6,
        min_alpha: float = 1e-6,
        min_beta: float = 1e-6,
        **layer_args,
    ) -> None:
        super().__init__(
            base_layer=base_layer,
            event_dim=event_dim,
            num_params=4,
            **layer_args,
        )

        self.min_lmbda = min_lmbda
        self.min_alpha = min_alpha
        self.min_beta = min_beta

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        x = super().forward(x)
        loc = x[:, : self.event_dim]
        lmbda = torch.clamp(
            F.softplus(x[:, self.event_dim : 2 * self.event_dim]), min=self.min_lmbda
        )
        alpha = 1 + torch.clamp(
            F.softplus(x[:, 2 * self.event_dim : 3 * self.event_dim]), min=self.min_alpha
        )
        beta = torch.clamp(F.softplus(x[:, 3 * self.event_dim :]), min=self.min_beta)
        return {
            "loc": loc,
            "lmbda": lmbda,
            "alpha": alpha,
            "beta": beta,
        }
