from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class SILog(Metric):
    def __init__(
        self, sqrt: bool = False, lmbda: float = 1.0, **kwargs: Any
    ) -> None:
        r"""The Scale-Invariant Logarithmic Loss metric.

        .. math:: \text{SILog} = \frac{1}{N} \sum_{i=1}^{N} \left(\log(y_i) - \log(\hat{y_i})\right)^2 - \left(\frac{1}{N} \sum_{i=1}^{N} \log(y_i) \right)^2,

        where :math:`N` is the batch size, :math:`y_i` is a tensor of target
        values and :math:`\hat{y_i}` is a tensor of prediction.
        Return the square root of SILog by setting :attr:`sqrt` to `True`.

        Args:
            sqrt: If `True`, return the square root of the metric. Defaults to
                False.
            lmbda: The regularization parameter on the variance of error.
                Defaults to 1.0.
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Reference:
            Depth Map Prediction from a Single Image using a Multi-Scale Deep Network. David Eigen, Christian Puhrsch, Rob Fergus. NeurIPS 2014.
            From Big to Small: Multi-Scale Local Planar Guidance for Monocular Depth Estimation. Jin Han Lee, Myung-Kyu Han, Dong Wook Ko and Il Hong Suh. (For :attr:`lmbda`)
        """
        super().__init__(**kwargs)
        self.sqrt = sqrt
        self.lmbda = lmbda
        self.add_state(
            "log_dists",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state(
            "sq_log_dists",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets.

        Args:
            pred (Tensor): A prediction tensor of shape (batch)
            target (Tensor): A tensor of ground truth labels of shape (batch)
        """
        self.log_dists += torch.sum(pred.log() - target.log())
        self.sq_log_dists += torch.sum((pred.log() - target.log()) ** 2)
        self.total += target.size(0)

    def compute(self) -> Tensor:
        """Compute the Scale-Invariant Logarithmic Loss."""
        log_dists = dim_zero_cat(self.log_dists)
        sq_log_dists = dim_zero_cat(self.sq_log_dists)
        out = sq_log_dists / self.total - self.lmbda * log_dists**2 / (
            self.total * self.total
        )
        if self.sqrt:
            return torch.sqrt(out)
        return out
