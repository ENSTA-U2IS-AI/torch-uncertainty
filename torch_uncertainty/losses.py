from typing import Literal

import torch
from torch import Tensor, nn
from torch.distributions import Distribution
from torch.nn import functional as F

from torch_uncertainty.layers.bayesian import bayesian_modules
from torch_uncertainty.utils.distributions import NormalInverseGamma


class DistributionNLLLoss(nn.Module):
    def __init__(
        self, reduction: Literal["mean", "sum"] | None = "mean"
    ) -> None:
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
            padding_mask (Tensor, optional): The padding mask. Defaults to None.
                Sets the loss to 0 for padded values.
        """
        loss = -dist.log_prob(targets)
        if padding_mask is not None:
            loss = loss.masked_fill(padding_mask, 0.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss


class KLDiv(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        """KL divergence loss for Bayesian Neural Networks. Gathers the KL from the
        modules computed in the forward passes.

        Args:
            model (nn.Module): Bayesian Neural Network
        """
        super().__init__()
        self.model = model

    def forward(self) -> Tensor:
        return self._kl_div()

    def _kl_div(self) -> Tensor:
        """Gathers pre-computed KL-Divergences from :attr:`model`."""
        kl_divergence = torch.zeros(1)
        count = 0
        for module in self.model.modules():
            if isinstance(module, bayesian_modules):
                kl_divergence = kl_divergence.to(
                    device=module.lvposterior.device
                )
                kl_divergence += module.lvposterior - module.lprior
                count += 1
        return kl_divergence / count


class ELBOLoss(nn.Module):
    def __init__(
        self,
        model: nn.Module | None,
        inner_loss: nn.Module,
        kl_weight: float,
        num_samples: int,
    ) -> None:
        """The Evidence Lower Bound (ELBO) loss for Bayesian Neural Networks.

        ELBO loss for Bayesian Neural Networks. Use this loss function with the
        objective that you seek to minimize as :attr:`inner_loss`.

        Args:
            model (nn.Module): The Bayesian Neural Network to compute the loss for
            inner_loss (nn.Module): The loss function to use during training
            kl_weight (float): The weight of the KL divergence term
            num_samples (int): The number of samples to use for the ELBO loss

        Note:
            Set the model to None if you use the ELBOLoss within
            the ClassificationRoutine. It will get filled automatically.
        """
        super().__init__()
        _elbo_loss_checks(inner_loss, kl_weight, num_samples)
        self.set_model(model)

        self.inner_loss = inner_loss
        self.kl_weight = kl_weight
        self.num_samples = num_samples

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Gather the KL divergence from the bayesian modules and aggregate
        the ELBO loss for a given network.

        Args:
            inputs (Tensor): The inputs of the Bayesian Neural Network
            targets (Tensor): The target values

        Returns:
            Tensor: The aggregated ELBO loss
        """
        aggregated_elbo = torch.zeros(1, device=inputs.device)
        for _ in range(self.num_samples):
            logits = self.model(inputs)
            aggregated_elbo += self.inner_loss(logits, targets)
            # TODO: This shouldn't be necessary
            aggregated_elbo += self.kl_weight * self._kl_div().to(inputs.device)
        return aggregated_elbo / self.num_samples

    def set_model(self, model: nn.Module | None) -> None:
        self.model = model
        if model is not None:
            self._kl_div = KLDiv(model)


def _elbo_loss_checks(
    inner_loss: nn.Module, kl_weight: float, num_samples: int
) -> None:
    if isinstance(inner_loss, type):
        raise TypeError(
            "The inner_loss should be an instance of a class."
            f"Got {inner_loss}."
        )

    if kl_weight < 0:
        raise ValueError(
            f"The KL weight should be non-negative. Got {kl_weight}."
        )

    if num_samples < 1:
        raise ValueError(
            "The number of samples should not be lower than 1."
            f"Got {num_samples}."
        )
    if not isinstance(num_samples, int):
        raise TypeError(
            "The number of samples should be an integer. "
            f"Got {type(num_samples)}."
        )


class DERLoss(DistributionNLLLoss):
    def __init__(
        self, reg_weight: float, reduction: str | None = "mean"
    ) -> None:
        """The Deep Evidential loss.

        This loss combines the negative log-likelihood loss of the normal
        inverse gamma distribution and a weighted regularization term.

        Args:
            reg_weight (float): The weight of the regularization term.
            reduction (str, optional): specifies the reduction to apply to the
            output:``'none'`` | ``'mean'`` | ``'sum'``.

        Reference:
            Amini, A., Schwarting, W., Soleimany, A., & Rus, D. (2019). Deep
            evidential regression. https://arxiv.org/abs/1910.02600.
        """
        super().__init__(reduction=None)

        if reduction not in (None, "none", "mean", "sum"):
            raise ValueError(f"{reduction} is not a valid value for reduction.")
        self.final_reduction = reduction

        if reg_weight < 0:
            raise ValueError(
                "The regularization weight should be non-negative, but got "
                f"{reg_weight}."
            )
        self.reg_weight = reg_weight

    def _reg(self, dist: NormalInverseGamma, targets: Tensor) -> Tensor:
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

        if self.final_reduction == "mean":
            return loss.mean()
        if self.final_reduction == "sum":
            return loss.sum()
        return loss


class BetaNLL(nn.Module):
    def __init__(
        self, beta: float = 0.5, reduction: str | None = "mean"
    ) -> None:
        """The Beta Negative Log-likelihood loss.

        Args:
            beta (float): TParameter from range [0, 1] controlling relative
            weighting between data points, where `0` corresponds to
            high weight on low error points and `1` to an equal weighting.
            reduction (str, optional): specifies the reduction to apply to the
            output:``'none'`` | ``'mean'`` | ``'sum'``.

        Reference:
            Seitzer, M., Tavakoli, A., Antic, D., & Martius, G. (2022). On the
            pitfalls of heteroscedastic uncertainty estimation with probabilistic
            neural networks. https://arxiv.org/abs/2203.09168.
        """
        super().__init__()

        if beta < 0 or beta > 1:
            raise ValueError(
                "The beta parameter should be in range [0, 1], but got "
                f"{beta}."
            )
        self.beta = beta
        self.nll_loss = nn.GaussianNLLLoss(reduction="none")
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"{reduction} is not a valid value for reduction.")
        self.reduction = reduction

    def forward(
        self, mean: Tensor, targets: Tensor, variance: Tensor
    ) -> Tensor:
        loss = self.nll_loss(mean, targets, variance) * (
            variance.detach() ** self.beta
        )

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class DECLoss(nn.Module):
    def __init__(
        self,
        annealing_step: int | None = None,
        reg_weight: float | None = None,
        loss_type: str = "log",
        reduction: str | None = "mean",
    ) -> None:
        """The deep evidential classification loss.

        Args:
            annealing_step (int): Annealing step for the weight of the
            regularization term.
            reg_weight (float): Fixed weight of the regularization term.
            loss_type (str, optional): Specifies the loss type to apply to the
            Dirichlet parameters: ``'mse'`` | ``'log'`` | ``'digamma'``.
            reduction (str, optional): Specifies the reduction to apply to the
            output:``'none'`` | ``'mean'`` | ``'sum'``.

        Reference:
            Sensoy, M., Kaplan, L., & Kandemir, M. (2018). Evidential deep
            learning to quantify classification uncertainty. NeurIPS 2018.
            https://arxiv.org/abs/1806.01768.
        """
        super().__init__()

        if reg_weight is not None and (reg_weight < 0):
            raise ValueError(
                "The regularization weight should be non-negative, but got "
                f"{reg_weight}."
            )
        self.reg_weight = reg_weight

        if annealing_step is not None and (annealing_step <= 0):
            raise ValueError(
                "The annealing step should be positive, but got "
                f"{annealing_step}."
            )
        self.annealing_step = annealing_step

        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"{reduction} is not a valid value for reduction.")
        self.reduction = reduction

        if loss_type not in ["mse", "log", "digamma"]:
            raise ValueError(
                f"{loss_type} is not a valid value for mse/log/digamma loss."
            )
        self.loss_type = loss_type

    def _mse_loss(self, evidence: Tensor, targets: Tensor) -> Tensor:
        evidence = torch.relu(evidence)
        alpha = evidence + 1.0
        strength = torch.sum(alpha, dim=1, keepdim=True)
        loglikelihood_err = torch.sum(
            (targets - (alpha / strength)) ** 2, dim=1, keepdim=True
        )
        loglikelihood_var = torch.sum(
            alpha * (strength - alpha) / (strength * strength * (strength + 1)),
            dim=1,
            keepdim=True,
        )
        return loglikelihood_err + loglikelihood_var

    def _log_loss(self, evidence: Tensor, targets: Tensor) -> Tensor:
        evidence = torch.relu(evidence)
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1, keepdim=True)
        return torch.sum(
            targets * (torch.log(strength) - torch.log(alpha)),
            dim=1,
            keepdim=True,
        )

    def _digamma_loss(self, evidence: Tensor, targets: Tensor) -> Tensor:
        evidence = torch.relu(evidence)
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1, keepdim=True)
        return torch.sum(
            targets * (torch.digamma(strength) - torch.digamma(alpha)),
            dim=1,
            keepdim=True,
        )

    def _kldiv_reg(
        self,
        evidence: Tensor,
        targets: Tensor,
    ) -> Tensor:
        num_classes = evidence.size()[-1]
        evidence = torch.relu(evidence)
        alpha = evidence + 1.0

        kl_alpha = (alpha - 1) * (1 - targets) + 1

        ones = torch.ones(
            [1, num_classes], dtype=evidence.dtype, device=evidence.device
        )
        sum_kl_alpha = torch.sum(kl_alpha, dim=1, keepdim=True)
        first_term = (
            torch.lgamma(sum_kl_alpha)
            - torch.lgamma(kl_alpha).sum(dim=1, keepdim=True)
            + torch.lgamma(ones).sum(dim=1, keepdim=True)
            - torch.lgamma(ones.sum(dim=1, keepdim=True))
        )
        second_term = torch.sum(
            (kl_alpha - ones)
            * (torch.digamma(kl_alpha) - torch.digamma(sum_kl_alpha)),
            dim=1,
            keepdim=True,
        )
        return first_term + second_term

    def forward(
        self,
        evidence: Tensor,
        targets: Tensor,
        current_epoch: int | None = None,
    ) -> Tensor:
        if (
            self.annealing_step is not None
            and self.annealing_step > 0
            and current_epoch is None
        ):
            raise ValueError(
                "The epoch num should be positive when \
                annealing_step is settled, but got "
                f"{current_epoch}."
            )

        if targets.ndim != 1:  # if no mixup or cutmix
            raise NotImplementedError(
                "DECLoss does not yet support mixup/cutmix."
            )
        # TODO: handle binary
        targets = F.one_hot(targets, num_classes=evidence.size()[-1])

        if self.loss_type == "mse":
            loss_dirichlet = self._mse_loss(evidence, targets)
        elif self.loss_type == "log":
            loss_dirichlet = self._log_loss(evidence, targets)
        else:  # self.loss_type == "digamma"
            loss_dirichlet = self._digamma_loss(evidence, targets)

        if self.reg_weight is None and self.annealing_step is None:
            annealing_coef = 0
        elif self.annealing_step is None and self.reg_weight > 0:
            annealing_coef = self.reg_weight
        else:
            annealing_coef = torch.min(
                torch.tensor(1.0, dtype=evidence.dtype),
                torch.tensor(
                    current_epoch / self.annealing_step, dtype=evidence.dtype
                ),
            )

        loss_reg = self._kldiv_reg(evidence, targets)
        loss = loss_dirichlet + annealing_coef * loss_reg
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        return loss
