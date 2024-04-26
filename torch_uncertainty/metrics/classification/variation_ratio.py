from typing import Literal

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class VariationRatio(Metric):
    full_state_update: bool = False
    is_differentiable: bool = True
    higher_is_better: bool = False

    def __init__(
        self,
        probabilistic: bool = True,
        reduction: Literal["mean", "sum", "none", None] = "mean",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        allowed_reduction = ("sum", "mean", "none", None)
        if reduction not in allowed_reduction:
            raise ValueError(
                "Expected argument `reduction` to be one of ",
                f"{allowed_reduction} but got {reduction}",
            )

        self.probabilistic = probabilistic
        self.reduction = reduction

        self.add_state("probs", [], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `VariationRatio` will save all predictions in buffer. For "
            " large datasets this may lead to large memory footprint."
        )

    def update(self, probs: Tensor) -> None:
        # store data as (example, estimator, class)
        self.probs.append(probs.transpose(0, 1))

    def compute(self) -> Tensor:
        r"""Computes the variation ratio which amounts to the proportion of
        predicted class labels which are not the chosen class.

        Returns:
            Tensor: Mean disagreement between estimators.
        """
        probs_per_est = dim_zero_cat(self.probs)
        n_estimators = probs_per_est.shape[1]
        probs = probs_per_est.mean(dim=1)

        # best class for example
        max_classes = probs.argmax(dim=-1)

        if self.probabilistic:
            probs_per_est = probs_per_est.permute((0, 2, 1))
            variation_ratio = 1 - probs_per_est[
                torch.arange(probs_per_est.size(0)), max_classes
            ].mean(dim=1)
        else:
            # best class for (example, estimator)
            max_classes_per_est = probs_per_est.argmax(dim=-1)
            variation_ratio = (
                1
                - torch.sum(
                    max_classes_per_est == max_classes.unsqueeze(1), dim=-1
                )
                / n_estimators
            )

        if self.reduction == "mean":
            variation_ratio = variation_ratio.mean()
        elif self.reduction == "sum":
            variation_ratio = variation_ratio.sum()
        else:  # None
            pass

        return variation_ratio
