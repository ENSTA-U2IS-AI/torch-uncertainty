import torch
from torch import Tensor
from torchmetrics import MeanAbsoluteError, MeanSquaredError


class MeanGTRelativeAbsoluteError(MeanAbsoluteError):
    def __init__(self, **kwargs) -> None:
        r"""Compute Mean Absolute Error relative to the Ground Truth (MAErel
            or ARErel).

        .. math:: \text{MAErel} = \frac{1}{N}\sum_i^N \frac{| y_i - \hat{y_i} |}{y_i}

        where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
            tensor of predictions.

        As input to ``forward`` and ``update`` the metric accepts the following
            input:

        - ``preds`` (:class:`~torch.Tensor`): Predictions from model
        - ``target`` (:class:`~torch.Tensor`): Ground truth values

        As output of ``forward`` and ``compute`` the metric returns the
            following output:

        - ``rel_mean_absolute_error`` (:class:`~torch.Tensor`): A tensor with
            the relative mean absolute error over the state

        Args:
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Reference:
            As in e.g. From big to small: Multi-scale local planar guidance for
            monocular depth estimation
        """
        super().__init__(**kwargs)

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        return super().update(pred / target, torch.ones_like(target))


class MeanGTRelativeSquaredError(MeanSquaredError):
    def __init__(
        self, squared: bool = True, num_outputs: int = 1, **kwargs
    ) -> None:
        r"""Compute mean squared error relative to the Ground Truth (MSErel or SRE).

        .. math:: \text{MSErel} = \frac{1}{N}\sum_i^N \frac{(y_i - \hat{y_i})^2}{y_i}

        Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
            tensor of predictions.

        As input to ``forward`` and ``update`` the metric accepts the following
            input:

        - ``preds`` (:class:`~torch.Tensor`): Predictions from model
        - ``target`` (:class:`~torch.Tensor`): Ground truth values

        As output of ``forward`` and ``compute`` the metric returns the
            following output:

        - ``rel_mean_squared_error`` (:class:`~torch.Tensor`): A tensor with
            the relative mean squared error

        Args:
            squared: If True returns MSErel value, if False returns RMSErel
                value.
            num_outputs: Number of outputs in multioutput setting
            kwargs: Additional keyword arguments, see `Advanced metric settings
                <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Reference:
            As in e.g. From big to small: Multi-scale local planar guidance for
                monocular depth estimation
        """
        super().__init__(squared, num_outputs, **kwargs)

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        return super().update(pred / torch.sqrt(target), torch.sqrt(target))
