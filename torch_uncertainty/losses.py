# fmt:off
from typing import Optional

import torch
from torch import Tensor, nn

from .layers.bayesian import bayesian_modules


# fmt: on
class KLDiv(nn.Module):
    """KL divergence loss for Bayesian Neural Networks. Gathers the KL from the
    modules computed in the forward passes.

    Args:
        model (nn.Module): Bayesian Neural Network
    """

    def __init__(self, model: nn.Module) -> None:
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
    """ELBO loss for Bayesian Neural Networks. Use this loss function with the
    objective that you seek to minimize as :attr:`criterion`.

    Args:
        model (nn.Module): The Bayesian Neural Network to compute the loss for
        criterion (nn.Module): The loss function to use during training
        kl_weight (float): The weight of the KL divergence term
        num_samples (int): The number of samples to use for the ELBO loss
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        kl_weight: float,
        num_samples: int,
    ) -> None:
        super().__init__()
        self.model = model
        self._kl_div = KLDiv(model)
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
    """The Normal Inverse-Gamma loss.

    Args:
        reg_weight (float): The weight of the regularization term.
        reduction (str, optional): specifies the reduction to apply to the
        output:``'none'`` | ``'mean'`` | ``'sum'``.

    Reference:
        Amini, A., Schwarting, W., Soleimany, A., & Rus, D. (2019). Deep
        evidential regression. https://arxiv.org/abs/1910.02600.
    """

    def __init__(
        self, reg_weight: float, reduction: Optional[str] = "mean"
    ) -> None:
        super().__init__()

        if reg_weight < 0:
            raise ValueError(
                "The regularization weight should be non-negative, but got "
                f"{reg_weight}."
            )
        self.reg_weight = reg_weight
        if reduction != "none" and reduction != "mean" and reduction != "sum":
            raise ValueError(f"{reduction} is not a valid value for reduction.")
        self.reduction = reduction

    def _nig_nll(self, gamma, v, alpha, beta, targets):
        Gamma = 2 * beta * (1 + v)
        nll = (
            0.5 * torch.log(torch.pi / v)
            - alpha * Gamma.log()
            + (alpha + 0.5) * torch.log(Gamma + v * (targets - gamma) ** 2)
            + torch.lgamma(alpha)
            - torch.lgamma(alpha + 0.5)
        )
        return nll

    def _nig_reg(self, gamma, v, alpha, targets):
        reg = torch.norm(targets - gamma, 1, dim=1, keepdim=True) * (
            2 * v + alpha
        )
        return reg

    def forward(self, gamma, v, alpha, beta, targets):
        loss_nll = self._nig_nll(gamma, v, alpha, beta, targets)
        loss_reg = self._nig_reg(gamma, v, alpha, targets)
        loss = loss_nll + self.reg_weight * loss_reg

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
