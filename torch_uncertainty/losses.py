# fmt:off
from torch import Tensor, nn

from .layers.bayesian_layers import (
    BayesConv1d,
    BayesConv2d,
    BayesConv3d,
    BayesLinear,
)

# fmt: on
bayesian_modules = (BayesConv1d, BayesConv2d, BayesConv3d, BayesLinear)


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
