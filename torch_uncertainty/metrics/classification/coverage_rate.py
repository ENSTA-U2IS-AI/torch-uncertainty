import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.compute import _safe_divide
from torchmetrics.utilities.imports import _XLA_AVAILABLE


class CoverageRate(Metric):
    is_differentiable = False
    higher_is_better = True
    full_state_update = False

    def __init__(
        self,
        num_classes: int | None = None,
        average: str = "micro",
        validate_args: bool = True,
        **kwargs,
    ):
        """Empirical coverage rate metric.

        Args:
            num_classes (int | None, optional): Number of classes. Defaults to ``None``.
            average (str, optional): Defines the reduction that is applied over labels. Should be
                one of the following:

                - ``'macro'`` (default): Compute the metric for each class separately and find their
                  unweighted mean. This does not take label imbalance into account.
                - ``'micro'``: Sum statistics across over all labels.

            validate_args (bool, optional): Whether to validate the arguments. Defaults to ``True``.
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.


        Raises:
            ValueError: If `num_classes` is `None` and `average` is not `micro`.
            ValueError: If `num_classes` is not an integer larger than 1.
            ValueError: If `average` is not one of `macro` or `micro`.
        """
        super().__init__(**kwargs)

        if validate_args:
            if num_classes is None and average != "micro":
                raise ValueError(
                    f"Argument `num_classes` can only be `None` for `average='micro'`, but got `average={average}`."
                )
            if num_classes is not None and (not isinstance(num_classes, int) or num_classes < 2):
                raise ValueError(
                    f"Expected argument `num_classes` to be an integer larger than 1, but got {num_classes}"
                )
            if average not in ["macro", "micro"]:
                raise ValueError("average must be either 'macro' or 'micro'.")

        self.num_classes = num_classes
        self.average = average
        self.validate_args = validate_args

        size = 1 if (average == "micro" or num_classes is None) else num_classes

        self.add_state("correct", torch.zeros(size, dtype=torch.long), dist_reduce_fx="sum")
        self.add_state("total", torch.zeros(size, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor) -> None:
        """Update the metric state with predictions and targets.

        Args:
            preds (torch.Tensor): predicted sets tensor of shape (B, C), where B is the batch size
                and C is the number of classes.
            target (torch.Tensor): target sets tensor of shape (B,).
        """
        batch_size = preds.size(0)
        target = target.long()

        covered = preds[torch.arange(batch_size), target]  # (B,)

        if self.average == "micro":
            self.correct += covered.sum()
            self.total += batch_size

        else:
            self.correct += _bincount(target[covered.bool()], self.num_classes)
            self.total += _bincount(target, self.num_classes)

    def compute(self) -> Tensor:
        """Compute the coverage rate.

        Returns:
            Tensor: The coverage rate.
        """
        if self.average == "micro":
            return _safe_divide(self.correct, self.total)
        return _safe_divide(self.correct, self.total).mean()


def _bincount(x: Tensor, minlength: int | None = None) -> Tensor:
    """Implement custom bincount.

    PyTorch currently does not support ``torch.bincount`` when running in deterministic mode on GPU or when running
    MPS devices or when running on XLA device. This implementation therefore falls back to using a combination of
    `torch.arange` and `torch.eq` in these scenarios. A small performance hit can expected and higher memory consumption
    as `[batch_size, mincount]` tensor needs to be initialized compared to native ``torch.bincount``.

    Args:
        x: tensor to count
        minlength: minimum length to count

    Returns:
        Number of occurrences for each unique element in x

    Example:
        >>> x = torch.tensor([0,0,0,1,1,2,2,2,2])
        >>> _bincount(x, minlength=3)
        tensor([3, 2, 4])

    Source:
        https://github.com/Lightning-AI/torchmetrics/blob/master/src/torchmetrics/utilities/data.py#L178

    """
    if minlength is None:
        minlength = len(torch.unique(x))

    if torch.are_deterministic_algorithms_enabled() or _XLA_AVAILABLE or x.is_mps:
        mesh = torch.arange(minlength, device=x.device).repeat(len(x), 1)
        return torch.eq(x.reshape(-1, 1), mesh).sum(dim=0)

    return torch.bincount(x, minlength=minlength)
