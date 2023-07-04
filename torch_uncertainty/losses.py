# fmt:off
import torch
from torch import Tensor, nn

from .layers.bayesian_layers import bayesian_modules


# fmt: on
class KL_Loss(nn.Module):
    """KL divergence loss for Bayesian Neural Networks

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


class ELBO_Loss(nn.Module):
    """ELBO loss for Bayesian Neural Networks

    Args:
        model (nn.Module): Bayesian Neural Network

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
        self._kl_div = KL_Loss(model)
        self.criterion = criterion
        self.kl_weight = kl_weight
        self.num_samples = num_samples

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        aggregated_elbo = 0
        for _ in range(self.num_samples):
            loss = self.criterion(logits, targets)
            aggregated_elbo += loss + self.kl_weight * self._kl_div()
        return aggregated_elbo / self.num_samples
