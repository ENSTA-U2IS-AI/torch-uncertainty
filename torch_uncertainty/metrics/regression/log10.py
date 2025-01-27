import torch
from torch import Tensor
from torchmetrics import MeanAbsoluteError


class Log10(MeanAbsoluteError):
    def __init__(self, **kwargs) -> None:
        r"""Computes the LOG10 metric.

        The Log10 metric computes the mean absolute error in the base-10 logarithmic space.

        .. math:: \text{Log10} = \frac{1}{N} \sum_{i=1}^{N} |\log_{10}(y_i) - \log_{10}(\hat{y_i})|

        where:
        - :math:`N` is the number of elements in the batch.
        - :math:`y_i` represents the true target values.
        - :math:`\hat{y_i}` represents the predicted values.

        This metric is useful for scenarios where the data spans multiple orders of magnitude, and evaluating
        error in log-space provides a more meaningful comparison.

        Inputs:
        - :attr:`preds`: :math:`(N)`
        - :attr:`target`: :math:`(N)`

        Args:
            kwargs: Additional keyword arguments, see `Advanced metric settings <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Example:

        .. code-block:: python

            from torch_uncertainty.metrics.regression import Log10
            import torch

            # Initialize the metric
            log10_metric = Log10()

            # Example predictions and targets
            preds = torch.tensor([10.0, 100.0, 1000.0])
            target = torch.tensor([12.0, 95.0, 1020.0])

            # Update the metric state
            log10_metric.update(preds, target)

            # Compute the Log10 error
            result = log10_metric.compute()
            print(f"Log10 Error: {result.item()}")
            # Output: Log10 Error: 0.03668594
        """
        super().__init__(**kwargs)
        self.add_state("values", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        return super().update(pred.log10(), target.log10())
