# fmt:off
from typing import Any, Literal, Optional

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


# fmt:on
class MutualInformation(Metric):
    """The Mutual Information Metric to estimate the epistemic uncertainty of
    an ensemble of estimators.

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
        A higher mutual information means a higher uncertainty.

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
                "values", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
        else:
            self.add_state("values", default=[], dist_reduce_fx="cat")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, probs: torch.Tensor) -> None:
        """Update the current mutual information with a new tensor of
        probabilities.

        Args:
            probs (torch.Tensor): Probabilities from the ensemble.
        """
        batch_size = probs.size(0)

        ens_probs = probs.mean(dim=1)
        entropy_mean = torch.special.entr(ens_probs).sum(dim=-1)
        mean_entropy = torch.special.entr(probs).sum(dim=-1).mean(dim=1)

        mutual_information = entropy_mean - mean_entropy

        if self.reduction is None or self.reduction == "none":
            self.values.append(mutual_information)
        else:
            self.values += mutual_information.sum()
            self.total += batch_size

    def compute(self) -> torch.Tensor:
        """Computes Mutual Information based on inputs passed in to ``update``
        previously.
        """
        values = dim_zero_cat(self.values)
        if self.reduction == "sum":
            return values.sum(dim=-1)
        elif self.reduction == "mean":
            return values.sum(dim=-1) / self.total
        else:  # reduction is None or "none"
            return values


class MutualInformation_old(Metric):
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


if __name__ == "__main__":
    input = torch.softmax(torch.randn(8, 5, 4), dim=-1)  # (B, N, C)

    mi_old = MutualInformation_old()
    mi = MutualInformation()

    assert mi_old(input.transpose(0, 1)) == mi(input)

    print("All good!")
