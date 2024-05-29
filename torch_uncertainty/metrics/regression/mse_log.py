from torch import Tensor
from torchmetrics import MeanSquaredError


class MeanSquaredLogError(MeanSquaredError):
    def __init__(self, squared: bool = True, **kwargs) -> None:
        r"""MeanSquaredLogError (MSELog) regression metric.

        .. math:: \text{MSELog} = \frac{1}{N}\sum_i^N  (\log \hat{y_i} - \log y_i)^2

        where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
            tensor of predictions.

        As input to ``forward`` and ``update`` the metric accepts the following
            input:

        - ``preds`` (:class:`~torch.Tensor`): Predictions from model
        - ``target`` (:class:`~torch.Tensor`): Ground truth values

        As output of ``forward`` and ``compute`` the metric returns the
            following output:

        - ``mse_log`` (:class:`~torch.Tensor`): A tensor with the
            relative mean absolute error over the state

        Args:
            squared: If True returns MSELog value, if False returns EMSELog
                value.
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Reference:
            As in e.g. From big to small: Multi-scale local planar guidance for
                monocular depth estimation
        """
        super().__init__(squared, **kwargs)

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        return super().update(pred.log(), target.log())
