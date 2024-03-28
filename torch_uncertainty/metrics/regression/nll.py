from torch import Tensor, distributions
from torchmetrics.utilities.data import dim_zero_cat

from torch_uncertainty.metrics import CategoricalNLL


class DistributionNLL(CategoricalNLL):
    def update(self, dist: distributions.Distribution, target: Tensor) -> None:
        """Update state with the predicted distributions and the targets.

        Args:
            dist (torch.distributions.Distribution): Predicted distributions.
            target (Tensor): Ground truth labels.
        """
        if self.reduction is None or self.reduction == "none":
            self.values.append(-dist.log_prob(target))
        else:
            self.values += -dist.log_prob(target).sum()
            self.total += target.size(0)

    def compute(self) -> Tensor:
        """Computes NLL based on inputs passed in to ``update`` previously."""
        values = dim_zero_cat(self.values)

        if self.reduction == "sum":
            return values.sum(dim=-1)
        if self.reduction == "mean":
            return values.sum(dim=-1) / self.total
        # reduction is None or "none"
        return values
