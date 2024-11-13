from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class BrierScore(Metric):
    is_differentiable: bool = True
    higher_is_better: bool | None = False
    full_state_update: bool = False

    def __init__(
        self,
        num_classes: int,
        top_class: bool = False,
        reduction: Literal["mean", "sum", "none", None] = "mean",
        **kwargs,
    ) -> None:
        r"""The Brier Score Metric.

        Args:
            num_classes (int): Number of classes
            top_class (bool, optional): If true, compute the Brier score for the
                top class only. Defaults to False.
            reduction (str, optional): Determines how to reduce over the
                :math:`B`/batch dimension:

                - ``'mean'`` [default]: Averages score across samples
                - ``'sum'``: Sum score across samples
                - ``'none'`` or ``None``: Returns score per sample

            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Inputs:
            - :attr:`probs`: :math:`(B, C)` or :math:`(B, N, C)`
            - :attr:`target`: :math:`(B)` or :math:`(B, C)`

            where :math:`B` is the batch size, :math:`C` is the number of classes
            and :math:`N` is the number of estimators.

        Note:
            If :attr:`probs` is a 3d tensor, then the metric computes the mean of
            the Brier score over the estimators ie. :math:`t = \frac{1}{N}
            \sum_{i=0}^{N-1} BrierScore(probs[:,i,:], target)`.

        Warning:
            Make sure that the probabilities in :attr:`probs` are normalized to sum
            to one.

        Raises:
            ValueError:
                If :attr:`reduction` is not one of ``'mean'``, ``'sum'``,
                ``'none'`` or ``None``.
        """
        super().__init__(**kwargs)

        allowed_reduction = ("sum", "mean", "none", None)
        if reduction not in allowed_reduction:
            raise ValueError(
                "Expected argument `reduction` to be one of ",
                f"{allowed_reduction} but got {reduction}",
            )

        self.num_classes = num_classes
        self.top_class = top_class
        self.reduction = reduction
        self.num_estimators = 1

        if self.reduction in ["mean", "sum"]:
            self.add_state(
                "values", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
        else:
            self.add_state("values", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, probs: Tensor, target: Tensor) -> None:
        """Update the current Brier score with a new tensor of probabilities.

        Args:
            probs (Tensor): A probability tensor of shape
                (batch, num_estimators, num_classes) or
                (batch, num_classes)
            target (Tensor): A tensor of ground truth labels of shape
                (batch, num_classes) or (batch)
        """
        if target.ndim == 1 and self.num_classes > 1:
            target = F.one_hot(target, self.num_classes)

        if probs.ndim <= 2:
            batch_size = probs.size(0)
        elif probs.ndim == 3:
            batch_size = probs.size(0)
            self.num_estimators = probs.size(1)
            target = target.unsqueeze(1).repeat(1, self.num_estimators, 1)
        else:
            raise ValueError(
                f"Expected `probs` to be of shape (batch, num_classes) or "
                f"(batch, num_estimators, num_classes) but got {probs.shape}"
            )

        if self.top_class:
            probs, indices = probs.max(dim=-1)
            target = target.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
            brier_score = F.mse_loss(probs, target, reduction="none")
        else:
            brier_score = F.mse_loss(probs, target, reduction="none").sum(
                dim=-1
            )

        if self.reduction is None or self.reduction == "none":
            self.values.append(brier_score)
        else:
            self.values += brier_score.sum()
            self.total += batch_size

    def compute(self) -> Tensor:
        """Compute the final Brier score based on inputs passed to ``update``.

        Returns:
            Tensor: The final value(s) for the Brier score
        """
        values = dim_zero_cat(self.values)
        if self.reduction == "sum":
            return values.sum(dim=-1) / self.num_estimators
        if self.reduction == "mean":
            return values.sum(dim=-1) / self.total / self.num_estimators
        return values
