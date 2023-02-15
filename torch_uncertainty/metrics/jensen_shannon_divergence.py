# fmt:off
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


# fmt:on
class JensenShannonDivergence(Metric):
    """The Jensen Shannon Divergence Metric to estimate the epistemic
    uncertainty of an ensemble of estimators.

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
        A higher Jensen-Shannon divergence means a higher uncertainty.

    Warning:
        Make sure that the probabilities in :attr:`probs` are normalized to sum
        to one.

    Raises:
        ValueError:
            If :attr:`reduction` is not one of ``'mean'``, ``'sum'``,
            ``'none'`` or ``None``.
    """

    full_state_update: bool = False
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

    def update(self, probs: Tensor) -> None:  # type: ignore
        """Update state with new data.

        Args:
            probs (torch.Tensor): Probabilities from the model.
        """
        batch_size = probs.size(0)
        num_estimators = probs.size(1)
        mean_probs = probs.mean(dim=1, keepdim=True).repeat(
            1, probs.shape[1], 1
        )
        jsd = (
            F.kl_div(
                mean_probs.log(),
                probs.log(),
                log_target=True,
                reduction="none",
            ).sum(dim=(1, 2))
            / num_estimators
        )

        if self.reduction is None or self.reduction == "none":
            self.values.append(jsd)
        else:
            self.values += jsd.sum()
            self.total += batch_size

    def compute(self) -> Tensor:
        """Compute Jensen Shannon Divergence based on inputs passed in to
        ``update``.
        """
        values = dim_zero_cat(self.values)
        if self.reduction == "sum":
            return values.sum(dim=-1)
        elif self.reduction == "mean":
            return values.sum(dim=-1) / self.total
        else:  # reduction is None or "none"
            return values


class JensenShannonDivergence_old(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_state("probs", [], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `JensenShannonDivergence` will save all "
            "predictions in buffer. For large datasets this may lead to large "
            "memory footprint."
        )

    def update(self, probs: Tensor) -> None:  # type: ignore
        # store data as (example, estimator, class)
        self.probs.append(probs.transpose(0, 1))

    def compute(self) -> Tensor:
        probs = dim_zero_cat(self.probs)
        mean_proba = probs.mean(1, keepdim=True).repeat(1, probs.shape[1], 1)

        return (
            F.kl_div(
                mean_proba.log(),
                probs.log(),
                log_target=True,
                reduction="batchmean",
            )
            / probs.shape[1]
        )


if __name__ == "__main__":
    import pytest

    input = torch.softmax(torch.randn(8, 5, 4), dim=-1)  # (B, N, C)

    mi_old = JensenShannonDivergence_old()
    mi = JensenShannonDivergence()

    print(mi_old(input.transpose(0, 1)))
    print(mi(input))

    assert mi_old(input.transpose(0, 1)) == pytest.approx(mi(input))

    print("All good!")
