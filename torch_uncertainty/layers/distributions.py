import torch.nn.functional as F
from torch import Tensor, distributions, nn


class AbstractDistLayer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError(f"dim must be positive, got {dim}.")
        self.dim = dim

    def forward(self, x: Tensor) -> distributions.Distribution:
        raise NotImplementedError


class IndptNormalDistLayer(AbstractDistLayer):
    def __init__(self, dim: int, min_scale: float = 1e-3) -> None:
        super().__init__(dim)
        if min_scale <= 0:
            raise ValueError(f"min_scale must be positive, got {min_scale}.")
        self.min_scale = min_scale

    def forward(self, x: Tensor) -> distributions.Normal:
        """Forward pass of the independent normal distribution layer.

        Args:
            x (Tensor): The input tensor of shape (dx2).

        Returns:
            distributions.Normal: The independent normal distribution.
        """
        loc = x[:, : self.dim]
        scale = F.softplus(x[:, self.dim :]) + self.min_scale
        if self.dim == 1:
            loc = loc.squeeze(1)
            scale = scale.squeeze(1)
        return distributions.Normal(loc, scale)


class IndptLaplaceDistLayer(AbstractDistLayer):
    def __init__(self, dim: int, min_scale: float = 1e-3) -> None:
        super().__init__(dim)
        if min_scale <= 0:
            raise ValueError(f"min_scale must be positive, got {min_scale}.")
        self.min_scale = min_scale

    def forward(self, x: Tensor) -> distributions.Laplace:
        """Forward pass of the independent normal distribution layer.

        Args:
            x (Tensor): The input tensor of shape (dx2).

        Returns:
            distributions.Laplace: The independent Laplace distribution.
        """
        loc = x[:, : self.dim]
        scale = F.softplus(x[:, self.dim :]) + self.min_scale
        if self.dim == 1:
            loc = loc.squeeze(1)
            scale = scale.squeeze(1)
        return distributions.Laplace(loc, scale)
