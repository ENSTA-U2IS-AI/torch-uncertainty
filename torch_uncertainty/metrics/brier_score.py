# fmt: off
from typing import Literal

import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


# fmt:on
class BrierScore(Metric):
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
        self.num_estimators = 1

        if self.reduction in ["mean", "sum"]:
            self.add_state(
                "values", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
        else:
            self.add_state("values", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, probs: torch.Tensor, target: torch.Tensor) -> None:
        """
        Update the current Brier score with a new tensor of probabilities.

        Args:
            probs (torch.Tensor): A probability tensor of shape
                (num_estimators, batch, num_classes) or
                (batch, num_classes)
        """
        if len(probs.shape) == 2:
            batch_size = probs.size(0)
        else:
            batch_size = probs.size(1)
            self.num_estimators = probs.size(2)
            target = target.unsqueeze(0)

        brier_score = F.mse_loss(probs, target, reduce=False).sum(dim=-1)

        if self.reduction is None or self.reduction == "none":
            self.values.append(brier_score)
        else:
            self.values += brier_score.sum()
            self.total += batch_size

    def compute(self) -> torch.Tensor:
        """
        Compute the final Brier score based on inputs passed to ``update``.

        Returns:
            torch.Tensor: The final value(s) for the Brier score
        """
        values = dim_zero_cat(self.values)
        if self.reduction == "sum":
            return values.sum(dim=-1) / self.num_estimators
        elif self.reduction == "mean":
            return values.sum(dim=-1) / self.total / self.num_estimators
        else:  # reduction is None
            return values
