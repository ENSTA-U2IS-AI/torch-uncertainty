from typing import Any

import torch
from torch import Tensor
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat


class SILog(Metric):
    def __init__(self, **kwargs: Any) -> None:
        r"""The Scale-Invariant Logarithmic Loss metric.

        .. math:: \text{SILog} = \frac{1}{N} \sum_{i=1}^{N} \left(\log(y_i) - \log(\hat{y_i})\right)^2 - \left(\frac{1}{N} \sum_{i=1}^{N} \log(y_i) \right)^2

        where :math:`N` is the batch size, :math:`y_i` is a tensor of target values and :math:`\hat{y_i}` is a tensor of prediction.

        Inputs:
            - :attr:`pred`: :math:`(N)`
            - :attr:`target`: :math:`(N)`

            where :math:`N` is the batch size.

        Reference:
            Depth Map Prediction from a Single Image using a Multi-Scale Deep Network.
        """
        super().__init__(**kwargs)
        self.add_state("log_dists", default=[], dist_reduce_fx="cat")

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        self.log_dists.append(torch.flatten(pred.log() - target.log()))

    def compute(self) -> Tensor:
        """Compute the Scale-Invariant Logarithmic Loss."""
        log_dists = dim_zero_cat(self.log_dists)
        return torch.mean(log_dists**2) - torch.sum(log_dists) ** 2 / (
            log_dists.size(0) * log_dists.size(0)
        )
