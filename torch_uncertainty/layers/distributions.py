from abc import ABC, abstractmethod

import torch.nn.functional as F
from torch import Tensor, nn
from torch.distributions import Distribution, Laplace, Normal

from torch_uncertainty.utils.distributions import NormalInverseGamma


class TUDist(ABC, nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}.")
        self.dim = dim

    @abstractmethod
    def forward(self, x: Tensor) -> Distribution:
        pass


class NormalLayer(TUDist):
    """Normal distribution layer.

    Converts model outputs to Independent Normal distributions.

    Args:
        dim (int): The number of independent dimensions for each prediction.
        eps (float): The minimal value of the :attr:`scale` parameter.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__(dim)
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.eps = eps

    def forward(self, x: Tensor) -> Normal:
        r"""Forward pass of the Normal distribution layer.

        Args:
            x (Tensor): A tensor of shape (:attr:`dim` :math:`\times`2).

        Returns:
            Normal: The output normal distribution.
        """
        loc = x[:, : self.dim]
        scale = F.softplus(x[:, self.dim :]) + self.eps
        return Normal(loc, scale)


class LaplaceLayer(TUDist):
    """Laplace distribution layer.

    Converts model outputs to Independent Laplace distributions.

    Args:
        dim (int): The number of independent dimensions for each prediction.
        eps (float): The minimal value of the :attr:`scale` parameter.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__(dim)
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}.")
        self.eps = eps

    def forward(self, x: Tensor) -> Laplace:
        r"""Forward pass of the Laplace distribution layer.

        Args:
            x (Tensor): A tensor of shape (..., :attr:`dim` :math:`\times`2).

        Returns:
            Laplace: The output Laplace distribution.
        """
        loc = x[..., : self.dim]
        scale = F.softplus(x[..., self.dim :]) + self.eps
        return Laplace(loc, scale)


class NormalInverseGammaLayer(TUDist):
    """Normal-Inverse-Gamma distribution layer.

    Converts model outputs to Independent Normal-Inverse-Gamma distributions.

    Args:
        dim (int): The number of independent dimensions for each prediction.
        eps (float): The minimal values of the :attr:`lmbda`, :attr:`alpha`-1
            and :attr:`beta` parameters.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__(dim)
        self.eps = eps

    def forward(self, x: Tensor) -> NormalInverseGamma:
        r"""Forward pass of the NormalInverseGamma distribution layer.

        Args:
            x (Tensor): A tensor of shape (:attr:`dim` :math:`\times`4).

        Returns:
            NormalInverseGamma: The output NormalInverseGamma distribution.
        """
        loc = x[..., : self.dim]
        lmbda = F.softplus(x[..., self.dim : 2 * self.dim]) + self.eps
        alpha = 1 + F.softplus(x[..., 2 * self.dim : 3 * self.dim]) + self.eps
        beta = F.softplus(x[..., 3 * self.dim :]) + self.eps
        return NormalInverseGamma(loc, lmbda, alpha, beta)
