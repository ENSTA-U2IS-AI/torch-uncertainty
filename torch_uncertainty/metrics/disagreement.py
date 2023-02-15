# fmt: off
from typing import Any, List, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


# fmt: on
class Disagreement(Metric):
    """The Disagreement Metric to estimate the confidence of an ensemble of
    estimators.

    Args:
        reduction (str, optional): Determines how to reduce over the
            :math:`B`/batch dimension:

            - ``'mean'`` [default]: Averages score across samples
            - ``'sum'``: Sum score across samples
            - ``'none'`` or ``None``: Returns score per sample

        kwargs: Additional keyword arguments, see `Advanced metric settings
            <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

    Inputs:
        - :attr:`probs`: :math:`(B, N, C)`

        where :math:`B` is the batch size, :math:`C` is the number of classes
        and :math:`N` is the number of estimators.

    Note:
        A higher disagreement means a lower confidence.

    Warning:
        Make sure that the probabilities in :attr:`probs` are normalized to sum
        to one.

    Raises:
        ValueError:
            If :attr:`reduction` is not one of ``'mean'``, ``'sum'``,
            ``'none'`` or ``None``.
    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        reduction: Literal["mean", "sum", "none", None] = "mean",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        allowed_reduction = ("sum", "mean", "none", None)
        if reduction not in allowed_reduction:
            raise ValueError(
                "Expected argument `reduction` to be one of ",
                f"{allowed_reduction} but got {reduction}",
            )

        self.reduction = reduction

        if self.reduction in ["mean", "sum"]:
            self.add_state(
                "values",
                default=torch.tensor(0.0),
                dist_reduce_fx="sum",
            )
        else:
            self.add_state("values", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def _compute_disagreement(self, preds: Tensor) -> Tensor:
        num_estimators = preds.size(-1)
        counts = torch.sum(F.one_hot(preds), dim=1)
        max_counts = num_estimators * (num_estimators - 1) / 2
        return 1 - (counts * (counts - 1) / 2).sum(dim=1) / max_counts

    def update(self, probs: Tensor) -> None:
        """Update state with prediction probabilities and targets.

        Args:
            probs (torch.Tensor): Probabilities from the model.
        """
        preds = probs.argmax(dim=-1)
        if self.reduction is None or self.reduction == "none":
            self.values.append(self._compute_disagreement(preds))
        else:
            self.values += self._compute_disagreement(preds).sum(dim=-1)
            self.total += probs.size(0)

    def compute(self) -> Tensor:
        """Compute Disagreement based on inputs passed in to ``update``."""
        values = dim_zero_cat(self.values)
        if self.reduction == "sum":
            return values.sum(dim=-1)
        elif self.reduction == "mean":
            return values.sum(dim=-1) / self.total
        else:  # reduction is None or "none"
            return values


# TODO: Check dimension
class Disagreement_old(Metric):
    full_state_update: bool = False
    probs: List[Tensor]

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_state("probs", [], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `Disagreement` will save all targets and predictions"
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
        max_counts = n_estimators * (n_estimators - 1) / 2
        return torch.mean(
            1 - (counts * (counts - 1) / 2).sum(dim=1) / max_counts
        )

    def compute(self) -> Tensor:
        probs = dim_zero_cat(self.probs)
        classes = probs.argmax(dim=-1)
        return self._compute_disagreement(classes)


if __name__ == "__main__":
    input = torch.softmax(torch.randn(8, 5, 4), dim=-1)  # (B, N, C)

    disagreement_old = Disagreement_old()
    disagreement = Disagreement()

    assert disagreement_old(input.transpose(0, 1)) == disagreement(input)

    print("All good!")
