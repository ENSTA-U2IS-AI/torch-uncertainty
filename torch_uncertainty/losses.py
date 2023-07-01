# fmt:off
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

    def forward(self, input: Tensor, logits: Tensor) -> Tensor:
        return self._kl_div()

    def _kl_div(self) -> Tensor:
        """Gathers pre-computed KL-Divergences from :attr:`model`."""
        kl_divergence = 0
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
        self, criterion: nn.Module, model: nn.Module, num_samples: int
    ) -> None:
        super().__init__()
        self.criterion = criterion
        self.model = model
        self.num_samples = num_samples
        self._kl_div = KL_Loss(model)

    def forward(self, input: Tensor, logits: Tensor) -> Tensor:
        aggregated_elbo = 0
        for _ in range(self.num_samples):
            logits = self.model(input)
            loss = self.criterion(input, logits)
            aggregated_elbo += loss + self._kl_div()
        return self._kl_div()
