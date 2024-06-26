from numbers import Number

import torch
from einops import rearrange
from torch import Tensor
from torch.distributions import (
    Distribution,
    Laplace,
    Normal,
    constraints,
)
from torch.distributions.utils import broadcast_all


def dist_size(distribution: Distribution) -> torch.Size:
    """Get the size of the distribution.

    Args:
        distribution (Distribution): The distribution.

    Returns:
        torch.Size: The size of the distribution.
    """
    if isinstance(distribution, Normal | Laplace | NormalInverseGamma):
        return distribution.loc.size()
    raise NotImplementedError(
        f"Size of {type(distribution)} distributions is not supported."
        "Raise an issue if needed."
    )


def cat_dist(distributions: list[Distribution], dim: int) -> Distribution:
    """Concatenate a list of distributions into a single distribution.

    Args:
        distributions (list[Distribution]): The list of distributions.
        dim (int): The dimension to concatenate.

    Returns:
        Distribution: The concatenated distributions.
    """
    dist_type = type(distributions[0])
    if not all(
        isinstance(distribution, dist_type) for distribution in distributions
    ):
        raise ValueError("All distributions must have the same type.")

    if isinstance(distributions[0], Normal | Laplace):
        locs = torch.cat(
            [distribution.loc for distribution in distributions], dim=dim
        )
        scales = torch.cat(
            [distribution.scale for distribution in distributions], dim=dim
        )
        return dist_type(loc=locs, scale=scales)
    if isinstance(distributions[0], NormalInverseGamma):
        locs = torch.cat(
            [distribution.loc for distribution in distributions], dim=dim
        )
        lmbdas = torch.cat(
            [distribution.lmbda for distribution in distributions], dim=dim
        )
        alphas = torch.cat(
            [distribution.alpha for distribution in distributions], dim=dim
        )
        betas = torch.cat(
            [distribution.beta for distribution in distributions], dim=dim
        )
        return NormalInverseGamma(
            loc=locs, lmbda=lmbdas, alpha=alphas, beta=betas
        )
    raise NotImplementedError(
        f"Concatenation of {dist_type} distributions is not supported."
        "Raise an issue if needed."
    )


def dist_squeeze(distribution: Distribution, dim: int) -> Distribution:
    """Squeeze the distribution along a given dimension.

    Args:
        distribution (Distribution): The distribution to squeeze.
        dim (int): The dimension to squeeze.

    Returns:
        Distribution: The squeezed distribution.
    """
    dist_type = type(distribution)
    if isinstance(distribution, Normal | Laplace):
        loc = distribution.loc.squeeze(dim)
        scale = distribution.scale.squeeze(dim)
        return dist_type(loc=loc, scale=scale)
    if isinstance(distribution, NormalInverseGamma):
        loc = distribution.loc.squeeze(dim)
        lmbda = distribution.lmbda.squeeze(dim)
        alpha = distribution.alpha.squeeze(dim)
        beta = distribution.beta.squeeze(dim)
        return NormalInverseGamma(loc=loc, lmbda=lmbda, alpha=alpha, beta=beta)
    raise NotImplementedError(
        f"Squeezing of {dist_type} distributions is not supported."
        "Raise an issue if needed."
    )


def dist_rearrange(
    distribution: Distribution, pattern: str, **axes_lengths: int
) -> Distribution:
    dist_type = type(distribution)
    if isinstance(distribution, Normal | Laplace):
        loc = rearrange(distribution.loc, pattern=pattern, **axes_lengths)
        scale = rearrange(distribution.scale, pattern=pattern, **axes_lengths)
        return dist_type(loc=loc, scale=scale)
    if isinstance(distribution, NormalInverseGamma):
        loc = rearrange(distribution.loc, pattern=pattern, **axes_lengths)
        lmbda = rearrange(distribution.lmbda, pattern=pattern, **axes_lengths)
        alpha = rearrange(distribution.alpha, pattern=pattern, **axes_lengths)
        beta = rearrange(distribution.beta, pattern=pattern, **axes_lengths)
        return NormalInverseGamma(loc=loc, lmbda=lmbda, alpha=alpha, beta=beta)
    raise NotImplementedError(
        f"Rearrange of {dist_type} is not supported. Raise an issue if needed."
    )


class NormalInverseGamma(Distribution):
    arg_constraints = {
        "loc": constraints.real,
        "lmbda": constraints.positive,
        "alpha": constraints.greater_than_eq(1),
        "beta": constraints.positive,
    }
    support = constraints.real
    has_rsample = False

    def __init__(
        self,
        loc: Number | Tensor,
        lmbda: Number | Tensor,
        alpha: Number | Tensor,
        beta: Number | Tensor,
        validate_args: bool | None = None,
    ) -> None:
        self.loc, self.lmbda, self.alpha, self.beta = broadcast_all(
            loc, lmbda, alpha, beta
        )
        if (
            isinstance(loc, Number)
            and isinstance(lmbda, Number)
            and isinstance(alpha, Number)
            and isinstance(beta, Number)
        ):
            batch_shape = torch.Size()
        else:
            batch_shape = self.loc.size()
        super().__init__(batch_shape, validate_args=validate_args)

    @property
    def mean(self) -> Tensor:
        """Impromper mean of the NormalInverseGamma distribution.

        This value is necessary to perform point-wise predictions in the
        regression routine.
        """
        return self.loc

    def mode(self) -> None:
        raise NotImplementedError(
            "NormalInverseGamma distribution has no mode."
        )

    def stddev(self) -> None:
        raise NotImplementedError(
            "NormalInverseGamma distribution has no stddev."
        )

    def variance(self) -> None:
        raise NotImplementedError(
            "NormalInverseGamma distribution has no variance."
        )

    @property
    def mean_loc(self) -> Tensor:
        return self.loc

    @property
    def mean_variance(self) -> Tensor:
        return self.beta / (self.alpha - 1)

    @property
    def variance_loc(self) -> Tensor:
        return self.beta / (self.alpha - 1) / self.lmbda

    def log_prob(self, value: Tensor) -> Tensor:
        if self._validate_args:  # coverage: ignore
            self._validate_sample(value)
        gam: Tensor = 2 * self.beta * (1 + self.lmbda)
        return (
            -0.5 * torch.log(torch.pi / self.lmbda)
            + self.alpha * gam.log()
            - (self.alpha + 0.5)
            * torch.log(gam + self.lmbda * (value - self.loc) ** 2)
            - torch.lgamma(self.alpha)
            + torch.lgamma(self.alpha + 0.5)
        )
