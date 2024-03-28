import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class ThresholdAccuracy(Metric):
    def __init__(self, power: int, lmbda: float = 1.25, **kwargs) -> None:
        r"""The Threshold Accuracy metric, a.k.a. d1, d2, d3.

        Args:
            power: The power to raise the threshold to. Often in [1, 2, 3].
            lmbda: The threshold to compare the max of ratio of predictions
                to targets and its inverse to. Defaults to 1.25.
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.
        """
        super().__init__(**kwargs)
        if power < 0:
            raise ValueError(
                f"Power must be greater than or equal to 0. Got {power}."
            )
        self.power = power
        if lmbda < 1:
            raise ValueError(
                f"Lambda must be greater than or equal to 1. Got {lmbda}."
            )
        self.lmbda = lmbda
        self.add_state(
            "values", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        self.values += torch.sum(
            torch.max(preds / target, target / preds) < self.lmbda**self.power
        )
        self.total += target.size(0)

    def compute(self) -> Tensor:
        """Compute the Threshold Accuracy."""
        values = dim_zero_cat(self.values)
        return values / self.total
