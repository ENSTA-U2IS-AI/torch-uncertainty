# fmt:off
import torch
from torch import Tensor, nn

from .layers.bayesian_layers import bayesian_modules


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

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Gather the kl divergence from the bayesian modules and aggregate
        the ELBO loss for a given network.

        Args:
            logits (Tensor): The output of the Bayesian Neural Network
            targets (Tensor): The target values

        Returns:
            Tensor: The aggregated ELBO loss
        """
        aggregated_elbo = torch.zeros(1)
        for _ in range(self.num_samples):
            loss = self.criterion(logits, targets)
            aggregated_elbo += loss + self.kl_weight * self._kl_div()
        return aggregated_elbo / self.num_samples
