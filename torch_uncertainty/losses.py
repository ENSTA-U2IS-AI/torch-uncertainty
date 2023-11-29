import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .layers.bayesian import bayesian_modules


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
        for module in self.model.modules():
            if isinstance(module, bayesian_modules):
                kl_divergence = kl_divergence.to(
                    device=module.lvposterior.device
                )
                kl_divergence += module.lvposterior - module.lprior
        return kl_divergence


class ELBOLoss(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        kl_weight: float,
        num_samples: int,
    ) -> None:
        """The Evidence Lower Bound (ELBO) loss for Bayesian Neural Networks.

        ELBO loss for Bayesian Neural Networks. Use this loss function with the
        objective that you seek to minimize as :attr:`criterion`.

        Args:
            model (nn.Module): The Bayesian Neural Network to compute the loss for
            criterion (nn.Module): The loss function to use during training
            kl_weight (float): The weight of the KL divergence term
            num_samples (int): The number of samples to use for the ELBO loss
        """
        super().__init__()
        self.model = model
        self._kl_div = KLDiv(model)

        if isinstance(criterion, type):
            raise TypeError(
                "The criterion should be an instance of a class."
                f"Got {criterion}."
            )
        self.criterion = criterion

        if kl_weight < 0:
            raise ValueError(
                f"The KL weight should be non-negative. Got {kl_weight}."
            )
        self.kl_weight = kl_weight

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
        self.num_samples = num_samples

    def forward(self, inputs: Tensor, targets: Tensor) -> Tensor:
        """Gather the kl divergence from the bayesian modules and aggregate
        the ELBO loss for a given network.

        Args:
            inputs (Tensor): The *inputs* of the Bayesian Neural Network
            targets (Tensor): The target values

        Returns:
            Tensor: The aggregated ELBO loss
        """
        aggregated_elbo = torch.zeros(1, device=inputs.device)
        for _ in range(self.num_samples):
            logits = self.model(inputs)
            aggregated_elbo += self.criterion(logits, targets)
            aggregated_elbo += self.kl_weight * self._kl_div()
        return aggregated_elbo / self.num_samples


class NIGLoss(nn.Module):
    def __init__(
        self, reg_weight: float, reduction: str | None = "mean"
    ) -> None:
        """The Normal Inverse-Gamma loss.

        Args:
            reg_weight (float): The weight of the regularization term.
            reduction (str, optional): specifies the reduction to apply to the
            output:``'none'`` | ``'mean'`` | ``'sum'``.

        Reference:
            Amini, A., Schwarting, W., Soleimany, A., & Rus, D. (2019). Deep
            evidential regression. https://arxiv.org/abs/1910.02600.
        """
        super().__init__()

        if reg_weight < 0:
            raise ValueError(
                "The regularization weight should be non-negative, but got "
                f"{reg_weight}."
            )
        self.reg_weight = reg_weight
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"{reduction} is not a valid value for reduction.")
        self.reduction = reduction

    def _nig_nll(
        self,
        gamma: Tensor,
        v: Tensor,
        alpha: Tensor,
        beta: Tensor,
        targets: Tensor,
    ) -> Tensor:
        gam = 2 * beta * (1 + v)
        return (
            0.5 * torch.log(torch.pi / v)
            - alpha * gam.log()
            + (alpha + 0.5) * torch.log(gam + v * (targets - gamma) ** 2)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )

    def _nig_reg(
        self, gamma: Tensor, v: Tensor, alpha: Tensor, targets: Tensor
    ) -> Tensor:
        return torch.norm(targets - gamma, 1, dim=1, keepdim=True) * (
            2 * v + alpha
        )

    def forward(
        self,
        gamma: Tensor,
        v: Tensor,
        alpha: Tensor,
        beta: Tensor,
        targets: Tensor,
    ) -> Tensor:
        loss_nll = self._nig_nll(gamma, v, alpha, beta, targets)
        loss_reg = self._nig_reg(gamma, v, alpha, targets)
        loss = loss_nll + self.reg_weight * loss_reg

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
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
        # else:  # TODO: handle binary
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
