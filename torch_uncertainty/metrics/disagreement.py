from typing import List

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


class DisagreementMetric(Metric):
    full_state_update: bool = False
    probs: List[Tensor]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_state("probs", [], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `DisagreementMetric` will save all targets and predictions"
            " in buffer. For large datasets this may lead to large memory"
            " footprint."
        )

    def update(self, probs: Tensor) -> None:  # type: ignore
        # store data as (example, estimator, class)
        self.probs.append(probs.transpose(0, 1))

    def _compute_disagreement(self, classes: Tensor) -> Tensor:
        r"""Computes the disagreement between the predicted classes among
        all pairs of estimators.
        Args:
            classes (Tensor): Classes predicted by the `n_estimators`
                estimators.
        Returns:
            Tensor: Mean disagreement between estimators.
        """
        # TODO: Using onehot might be memory intensive
        n_estimators = classes.shape[-1]
        counts = torch.sum(F.one_hot(classes), dim=1)
        potential_counts = n_estimators * (n_estimators - 1) / 2
        return torch.mean(
            1 - (counts * (counts - 1) / 2).sum(dim=1) / potential_counts
        )

    def compute(self) -> Tensor:
        probs = dim_zero_cat(self.probs)
        classes = probs.argmax(dim=-1)
        return self._compute_disagreement(classes)
