import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Distribution, Laplace, Normal

from torch_uncertainty.utils.distributions import NormalInverseGamma


class AbstractDistLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}.")
        self.dim = dim

    def forward(self, x: Tensor) -> Distribution:
        raise NotImplementedError


class IndptNormalLayer(AbstractDistLayer):
    def __init__(self, dim: int, min_scale: float = 1e-6) -> None:
        super().__init__(dim)
        if min_scale <= 0:
            raise ValueError(f"min_scale must be positive, got {min_scale}.")
        self.min_scale = min_scale

    def forward(self, x: Tensor) -> Normal:
        """Forward pass of the independent normal distribution layer.

        Args:
            x (Tensor): The input tensor of shape (dx2).

        Returns:
            Normal: The independent normal distribution.
        """
        loc = x[:, : self.dim]
        scale = F.softplus(x[:, self.dim :]) + self.min_scale
        return Normal(loc, scale)


class IndptLaplaceLayer(AbstractDistLayer):
    def __init__(self, dim: int, min_scale: float = 1e-6) -> None:
        super().__init__(dim)
        if min_scale <= 0:
            raise ValueError(f"min_scale must be positive, got {min_scale}.")
        self.min_scale = min_scale

    def forward(self, x: Tensor) -> Laplace:
        """Forward pass of the independent Laplace distribution layer.

        Args:
            x (Tensor): The input tensor of shape (dx2).

        Returns:
            Laplace: The independent Laplace distribution.
        """
        loc = x[:, : self.dim]
        scale = F.softplus(x[:, self.dim :]) + self.min_scale
        return Laplace(loc, scale)


class IndptNormalInverseGammaLayer(AbstractDistLayer):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__(dim)
        self.eps = eps

    def forward(self, x: Tensor) -> Laplace:
        """Forward pass of the independent Laplace distribution layer.

        Args:
            x (Tensor): The input tensor of shape (dx2).

        Returns:
            Laplace: The independent Laplace distribution.
        """
        loc = x[:, : self.dim]
        lmbda = F.softplus(x[:, self.dim : 2 * self.dim]) + self.eps
        alpha = 1 + F.softplus(x[:, 2 * self.dim : 3 * self.dim]) + self.eps
        beta = F.softplus(x[:, 3 * self.dim :]) + self.eps
        return NormalInverseGamma(loc, lmbda, alpha, beta)
