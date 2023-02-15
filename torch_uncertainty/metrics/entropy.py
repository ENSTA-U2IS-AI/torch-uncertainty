from typing import Any, Literal, Optional

import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class Entropy(Metric):
    """The Shannon Entropy Metric to estimate the confidence of a single model
    or the mean confidence accross estimators.

    Args:
        reduction (str, optional): Determines how to reduce over the
            :math:`B`/batch dimension:

            - ``'mean'`` [default]: Averages score across samples
            - ``'sum'``: Sum score across samples
            - ``'none'`` or ``None``: Returns score per sample

        kwargs: Additional keyword arguments, see `Advanced metric settings
            <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

    Inputs:
        - ``probs``: :math:`(B, C)` or :math:`(B, N, C)`

        where :math:`B` is the batch size, :math:`C` is the number of classes
        and :math:`N` is the number of estimators.

    Note:
        A higher entropy means a lower confidence.

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
        """Update the current entropy with a new tensor of probabilities.

        Args:
            probs (torch.Tensor): Probabilities from the model.
        """
        batch_size = probs.size(0)
        entropy = torch.special.entr(probs).sum(dim=-1)

        if len(probs.shape) == 3:
            entropy = entropy.mean(dim=1)

        if self.reduction is None or self.reduction == "none":
            self.values.append(entropy)
        else:
            self.values += entropy.sum()
            self.total += batch_size

    def compute(self) -> torch.Tensor:
        """Computes Entropy based on inputs passed in to ``update``
        previously.
        """
        values = dim_zero_cat(self.values)
        if self.reduction == "sum":
            return values.sum(dim=-1)
        elif self.reduction == "mean":
            return values.sum(dim=-1) / self.total
        else:  # reduction is None or "none"
            return values


class Entropy_old(Metric):
    """The Shannon Entropy Metric to estimate the confidence of the
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
        - :attr:`probs`: :math:`(B, C)` or :math:`(N, B, C)`

        where :math:`B` is the batch size, :math:`C` is the number of classes
        and :math:`N` is the number of estimators.

    Note:
        A higher entropy means a lower confidence.

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
        """Update the current entropy with a new tensor of probabilities.

        Args:
            probs (torch.Tensor): Probabilities from the model.
        """
        if len(probs.shape) == 2:
            batch_size = probs.size(0)
        else:
            batch_size = probs.size(1)

        entropy = torch.special.entr(probs).sum(dim=-1)

        if len(probs.shape) == 3:
            entropy = entropy.mean(dim=0)

        if self.reduction is None or self.reduction == "none":
            self.values.append(entropy)
        else:
            self.values += entropy.sum()
            self.total += batch_size

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
    input = torch.softmax(torch.randn(8, 5, 4), dim=-1)  # (B, N, C)

    entropy_old = Entropy_old()
    entropy = Entropy()

    assert entropy_old(input.transpose(0, 1)) == entropy(input)

    input = torch.softmax(torch.randn(8, 4), dim=-1)  # (B, C)

    assert entropy_old(input) == entropy(input)

    print("All good!")
