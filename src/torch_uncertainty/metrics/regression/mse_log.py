from torch import Tensor
from torchmetrics import MeanSquaredError


class MeanSquaredLogError(MeanSquaredError):
    def __init__(self, squared: bool = True, **kwargs) -> None:
        r"""Computes the Mean Squared Logarithmic Error (MSLE) regression metric.

        This metric is commonly used in regression problems where the relative
        difference between predictions and targets is of greater importance than
        the absolute difference. It is particularly effective for datasets with
        wide-ranging magnitudes, as it penalizes underestimation more than
        overestimation.

        .. math:: \text{MSELog} = \frac{1}{N}\sum_i^N  (\log \hat{y_i} - \log y_i)^2

        where  :math:`y`  is a tensor of target values, and :math:`\hat{y}` is a
        tensor of predictions.

        As input to ``forward`` and ``update`` the metric accepts the following
        input:

        - **preds** (:class:`~torch.Tensor`): Predictions from model
        - **target** (:class:`~torch.Tensor`): Ground truth values

        As output of ``forward`` and ``compute`` the metric returns the
        following output:

        - **mse_log** (:class:`~torch.Tensor`): A tensor with the
          relative mean absolute error over the state

        Args:
            squared: If True returns MSELog value, if False returns EMSELog value.
            kwargs: Additional keyword arguments, see `Advanced metric settings <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Reference:
            [1] `From big to small: Multi-scale local planar guidance for monocular depth estimation
            <https://arxiv.org/abs/1907.10326>`_.

        Example:

        .. code-block:: python

            from torch_uncertainty.metrics.regression import MeanSquaredLogError
            import torch

            # Initialize the metric
            msle_metric = MeanSquaredLogError(squared=True)

            # Example predictions and targets (must be non-negative)
            preds = torch.tensor([2.5, 1.0, 2.0, 8.0])
            target = torch.tensor([3.0, 1.5, 2.0, 7.0])

            # Update the metric state
            msle_metric.update(preds, target)

            # Compute the Mean Squared Logarithmic Error
            result = msle_metric.compute()
            print(f"Mean Squared Logarithmic Error: {result.item()}")
            # Output: Mean Squared Logarithmic Error: 0.05386843904852867
        """
        super().__init__(squared, **kwargs)

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        return super().update(pred.log(), target.log())
