# fmt: off
from typing import Any, Literal, Optional

import torch
import torch.nn.functional as F
from torchmetrics import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat


# fmt: on
class NegativeLogLikelihood_old(Metric):
    full_state_update: bool = False

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

        self.add_state("log_probs", [], dist_reduce_fx="cat")
        self.add_state("targets", [], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `NLLMetric` will save all targets and predictions"
            " in buffer. For large datasets this may lead to large memory"
            " footprint."
        )

    def update(self, probs: torch.Tensor, target: torch.Tensor) -> None:
        self.log_probs.append(torch.log(probs))
        self.targets.append(target)

    def compute(self) -> torch.Tensor:
        log_probs = dim_zero_cat(self.log_probs)
        targets = dim_zero_cat(self.targets)
        return F.nll_loss(log_probs, targets)


class NegativeLogLikelihood(Metric):
    """The Negative Log Likelihood Metric.

    Args:
        reduction (str, optional): Determines how to reduce over the
            :math:`B`/batch dimension:

            - ``'mean'`` [default]: Averages score across samples
            - ``'sum'``: Sum score across samples
            - ``'none'`` or ``None``: Returns score per sample

        kwargs: Additional keyword arguments, see `Advanced metric settings
            <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

    Inputs:
        - :attr:`probs`: :math:`(B, C)`
        - :attr:`target`: :math:`(B)`

        where :math:`B` is the batch size and :math:`C` is the number of
        classes.

    Warning:
        Make sure that the probabilities in :attr:`probs` are normalized to sum
        to one.

    Raises:
        ValueError:
            If :attr:`reduction` is not one of ``'mean'``, ``'sum'``,
            ``'none'`` or ``None``.
    """

    is_differentiable: bool = False
    higher_is_better: Optional[bool] = False
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

    def update(self, probs: torch.Tensor, target: torch.Tensor) -> None:
        """Update state with prediction probabilities and targets.

        Args:
            probs (torch.Tensor): Probabilities from the model.
            target (torch.Tensor): Ground truth labels.
        """
        if self.reduction is None or self.reduction == "none":
            self.values.append(
                F.nll_loss(torch.log(probs), target, reduction="none")
            )
        else:
            self.values += F.nll_loss(torch.log(probs), target, reduction="sum")
            self.total += target.size(0)

    def compute(self) -> torch.Tensor:
        """Computes NLL based on inputs passed in to ``update`` previously."""
        values = dim_zero_cat(self.values)

        if self.reduction == "sum":
            return values.sum(dim=-1)
        elif self.reduction == "mean":
            return values.sum(dim=-1) / self.total
        else:  # reduction is None or "none"
            return values


if __name__ == "__main__":
    input = torch.softmax(torch.randn(8, 4), dim=-1)
    target = torch.randint(4, (8,))

    nll = NegativeLogLikelihood_old()
    nllv2 = NegativeLogLikelihood()

    assert nll(input, target) == nllv2(input, target)

    print("All good!")
