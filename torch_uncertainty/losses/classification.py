import torch
from torch import Tensor, nn
from torch.nn import functional as F


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

        if reduction not in ("none", "mean", "sum") and reduction is not None:
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


class ConfidencePenaltyLoss(nn.Module):
    def __init__(
        self,
        reg_weight: float = 1,
        reduction: str | None = "mean",
        eps: float = 1e-6,
    ) -> None:
        """The Confidence Penalty Loss.

        Args:
            reg_weight (float, optional): The weight of the regularization term.
            reduction (str, optional): specifies the reduction to apply to the
            output:``'none'`` | ``'mean'`` | ``'sum'``. Defaults to "mean".
            eps (float, optional): A small value to avoid numerical instability.
                Defaults to 1e-6.

        Reference:
            Gabriel Pereyra: Regularizing neural networks by penalizing
            confident output distributions. https://arxiv.org/pdf/1701.06548.

        """
        super().__init__()
        if reduction is None:
            reduction = "none"
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"{reduction} is not a valid value for reduction.")
        self.reduction = reduction
        if eps < 0:
            raise ValueError(
                "The epsilon value should be non-negative, but got " f"{eps}."
            )
        self.eps = eps
        if reg_weight < 0:
            raise ValueError(
                "The regularization weight should be non-negative, but got "
                f"{reg_weight}."
            )
        self.reg_weight = reg_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the Confidence Penalty loss.

        Args:
            logits (Tensor): The inputs of the Bayesian Neural Network
            targets (Tensor): The target values

        Returns:
            Tensor: The Confidence Penalty loss
        """
        probs = F.softmax(logits, dim=1)
        ce_loss = F.cross_entropy(logits, targets, reduction=self.reduction)
        reg_loss = torch.log(
            torch.tensor(logits.shape[-1], device=probs.device)
        ) + (probs * torch.log(probs + self.eps)).sum(dim=-1)
        if self.reduction == "sum":
            return ce_loss + self.reg_weight * reg_loss.sum()
        if self.reduction == "mean":
            return ce_loss + self.reg_weight * reg_loss.mean()
        return ce_loss + self.reg_weight * reg_loss


class ConflictualLoss(nn.Module):
    def __init__(
        self,
        reg_weight: float = 1,
        reduction: str | None = "mean",
    ) -> None:
        r"""The Conflictual Loss.

        Args:
            reg_weight (float, optional): The weight of the regularization term.
            reduction (str, optional): specifies the reduction to apply to the
            output:``'none'`` | ``'mean'`` | ``'sum'``.

        Reference:
            `Mohammed Fellaji et al. On the Calibration of Epistemic Uncertainty:
            Principles, Paradoxes and Conflictual Loss <https://arxiv.org/pdf/2407.12211>`_.
        """
        super().__init__()
        if reduction is None:
            reduction = "none"
        if reduction not in ("none", "mean", "sum"):
            raise ValueError(f"{reduction} is not a valid value for reduction.")
        self.reduction = reduction
        if reg_weight < 0:
            raise ValueError(
                "The regularization weight should be non-negative, but got "
                f"{reg_weight}."
            )
        self.reg_weight = reg_weight

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Compute the conflictual loss.

        Args:
            logits (Tensor): The outputs of the model.
            targets (Tensor): The target values.

        Returns:
            Tensor: The conflictual loss.
        """
        class_index = torch.randint(
            0, logits.shape[-1], (1,), dtype=torch.long, device=logits.device
        )
        ce_loss = F.cross_entropy(logits, targets, reduction=self.reduction)
        reg_loss = -F.log_softmax(logits, dim=1)[:, class_index]
        if self.reduction == "sum":
            return ce_loss + self.reg_weight * reg_loss.sum()
        if self.reduction == "mean":
            return ce_loss + self.reg_weight * reg_loss.mean()
        return ce_loss + self.reg_weight * reg_loss
