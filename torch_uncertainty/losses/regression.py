from typing import Literal

import torch
from torch import Tensor, nn
from torch.distributions import Distribution, Independent

from torch_uncertainty.utils.distributions import NormalInverseGamma


class DistributionNLLLoss(nn.Module):
    def __init__(self, reduction: Literal["mean", "sum"] | None = "mean") -> None:
        """Negative Log-Likelihood loss using given distributions as inputs.

        Args:
            reduction (str, optional): specifies the reduction to apply to the
                output:``'none'`` | ``'mean'`` | ``'sum'``. Defaults to "mean".
        """
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        dist: Distribution,
        targets: Tensor,
        padding_mask: Tensor | None = None,
    ) -> Tensor:
        """Compute the NLL of the targets given predicted distributions.

        Args:
            dist (Distribution): The predicted distributions
            targets (Tensor): The target values
            padding_mask (Tensor, optional): The padding mask. Defaults to ``None.``
                Sets the loss to ``0`` for padded values.
        """
        loss = -dist.log_prob(targets)
        if padding_mask is not None:
            loss = loss.masked_fill(padding_mask, float("nan"))

        if self.reduction == "mean":
            loss = loss.nanmean()
        elif self.reduction == "sum":
            loss = loss.nansum()
        return loss


class DERLoss(DistributionNLLLoss):
    def __init__(self, reg_weight: float, reduction: str | None = "mean") -> None:
        """The Deep Evidential Regression loss.

        This loss combines the negative log-likelihood loss of the normal
        inverse gamma distribution and a weighted regularization term.

        Args:
            reg_weight (float): The weight of the regularization term.
            reduction (str, optional): specifies the reduction to apply to the
                output:``'none'`` | ``'mean'`` | ``'sum'``.

        References:
            [1] `Amini, A., Schwarting, W., Soleimany, A., & Rus, D. (2019). Deep evidential regression
            <https://arxiv.org/abs/1910.02600>`_.

        """
        super().__init__(reduction=None)

        if reduction not in ("none", "mean", "sum") and reduction is not None:
            raise ValueError(f"{reduction} is not a valid value for reduction.")
        self.der_reduction = reduction

        if reg_weight < 0:
            raise ValueError(
                f"The regularization weight should be non-negative, but got {reg_weight}."
            )
        self.reg_weight = reg_weight

    def _reg(self, dist: NormalInverseGamma | Independent, targets: Tensor) -> Tensor:
        if isinstance(dist, Independent):
            dist = dist.base_dist

        return torch.norm(targets - dist.loc, 1, dim=1, keepdim=True) * (
            2 * dist.lmbda + dist.alpha
        )

    def forward(
        self,
        dist: NormalInverseGamma,
        targets: Tensor,
    ) -> Tensor:
        loss_nll = super().forward(dist, targets)
        loss_reg = self._reg(dist, targets)
        loss = loss_nll + self.reg_weight * loss_reg

        if self.der_reduction == "mean":
            return loss.mean()
        if self.der_reduction == "sum":
            return loss.sum()
        return loss


class BetaNLL(nn.Module):
    def __init__(self, beta: float = 0.5, reduction: str | None = "mean") -> None:
        """The Beta Negative Log-likelihood loss.

        Args:
            beta (float): Parameter from range [0, 1] controlling relative
                weighting between data points, where `0` corresponds to
                high weight on low error points and `1` to an equal weighting.
            reduction (str, optional): specifies the reduction to apply to the
                output:``'none'`` | ``'mean'`` | ``'sum'``.

        References:
            [1] `Seitzer, M., Tavakoli, A., Antic, D., & Martius, G. (2022). On the pitfalls of heteroscedastic uncertainty estimation with probabilistic neural networks
            <https://arxiv.org/abs/2203.09168>`_.

        """
        super().__init__()

        if beta < 0 or beta > 1:
            raise ValueError(f"The beta parameter should be in range [0, 1], but got {beta}.")
        self.beta = beta
        self.nll_loss = nn.GaussianNLLLoss(reduction="none")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"{reduction} is not a valid value for reduction.")
        self.reduction = reduction

    def forward(self, mean: Tensor, targets: Tensor, variance: Tensor) -> Tensor:
        loss = self.nll_loss(mean, targets, variance) * (variance.detach() ** self.beta)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss
