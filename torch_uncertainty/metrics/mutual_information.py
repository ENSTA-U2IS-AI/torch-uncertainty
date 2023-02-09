# fmt:off
from typing import Literal

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

    def __init__(
        self, reduction: Literal["mean", "sum", "none", None] = "mean", **kwargs
    ) -> None:
        super().__init__(**kwargs)

        allowed_reduction = ("sum", "mean", "none", None)
        if reduction not in allowed_reduction:
            raise ValueError(
                "Expected argument `reduction` to be one of ",
                f"{allowed_reduction} but got {reduction}",
            )

        self.reduction = reduction

        self.add_state("probs_per_est", [], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `MutualInformation` will save and predictions "
            "in buffer. For large datasets this may lead to a large memory "
            "footprint."
        )

    def update(self, probs_per_est: Tensor) -> None:  # type: ignore
        # store data as (example, estimator, class)
        if len(probs_per_est.shape) <= 2:
            raise ValueError("Please give the probabilities per estimator.")

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
        entropy_mean = torch.special.entr(probs).sum(dim=-1)

        # Mean over the estimators of the entropy over the classes
        mean_entropy = torch.special.entr(probs_per_est).sum(dim=-1).mean(dim=0)

        mutual_information: torch.Tensor = entropy_mean - mean_entropy

        if self.reduction == "mean":
            mutual_information = mutual_information.mean()
        elif self.reduction == "sum":
            mutual_information = mutual_information.sum()

        return mutual_information
