# fmt:off
import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


# fmt:on
class MutualInformation(Metric):
    r"""
    The Mutual Information to estimate the epistemic uncertainty.
    A higher mutual information means a higher uncertainty.
    """
    full_state_update: bool = False

    def __init__(self, reduction: str = "mean", **kwargs) -> None:
        super().__init__(**kwargs)

        self.reduction = reduction
        # self.entropy_over_estimators = Entropy(over_estimators=True)
        # self.entropy = Entropy()
        self.add_state("probs_per_est", [], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `MutualInformation` will save and predictions "
            "in buffer. For large datasets this may lead to a large memory "
            "footprint."
        )

    def update(self, probs_per_est: Tensor) -> None:  # type: ignore
        # store data as (example, estimator, class)
        self.probs_per_est.append(probs_per_est.transpose(0, 1))

    def compute(self) -> Tensor:
        r"""Computes the mutual information on the data.
        Returns:
            Tensor: The total mutual information.
        """

        # convert data to (estimator, example, class)
        probs_per_est = dim_zero_cat(self.probs_per_est).transpose(0, 1)
        probs = probs_per_est.mean(dim=0)

        # Entropy of the mean over the estimators
        entropy_product = torch.log(probs) * probs
        entropy_mean = -entropy_product.sum(dim=-1)

        # Mean over the estimators of the entropy over the classes
        entropy_product = torch.log(probs_per_est) * probs_per_est
        mean_entropy = (-entropy_product.sum(dim=-1)).mean(dim=0)

        mutual_information = entropy_mean - mean_entropy

        if self.reduction == "mean":
            mutual_information = mutual_information.mean()
        elif self.reduction == "sum":
            mutual_information = mutual_information.sum()

        return mutual_information
