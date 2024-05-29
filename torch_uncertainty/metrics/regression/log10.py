import torch
from torch import Tensor
from torchmetrics import MeanAbsoluteError


class Log10(MeanAbsoluteError):
    def __init__(self, **kwargs) -> None:
        r"""The Log10 metric.

        .. math:: \text{Log10} = \frac{1}{N} \sum_{i=1}^{N} |\log_{10}(y_i) - \log_{10}(\hat{y_i})|

        where :math:`N` is the batch size, :math:`y_i` is a tensor of target
            values and :math:`\hat{y_i}` is a tensor of prediction.

        Inputs:
            - :attr:`preds`: :math:`(N)`
            - :attr:`target`: :math:`(N)`

        Args:
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.
        """
        super().__init__(**kwargs)
        self.add_state(
            "values", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        return super().update(pred.log10(), target.log10())
