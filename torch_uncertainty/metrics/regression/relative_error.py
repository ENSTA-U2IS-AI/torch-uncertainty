import torch
from torch import Tensor
from torchmetrics import MeanAbsoluteError, MeanSquaredError


class MeanGTRelativeAbsoluteError(MeanAbsoluteError):
    def __init__(self, **kwargs) -> None:
        r"""Compute Mean Absolute Error relative to the Ground Truth (MAErel
        or ARErel).

        This metric is commonly used in tasks where the relative deviation of
        predictions with respect to the ground truth is important.

        .. math:: \text{MAErel} = \frac{1}{N}\sum_i^N \frac{| y_i - \hat{y_i} |}{y_i}

        where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
        tensor of predictions.

        As input to ``forward`` and ``update`` the metric accepts the following
        input:

        - **preds** (:class:`~torch.Tensor`): Predictions from model
        - **target** (:class:`~torch.Tensor`): Ground truth values

        As output of ``forward`` and ``compute`` the metric returns the
        following output:

        - **rel_mean_absolute_error** (:class:`~torch.Tensor`): A tensor with
          the relative mean absolute error over the state

        Args:
            kwargs: Additional keyword arguments, see `Advanced metric settings <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Reference:
            [1] `From big to small: Multi-scale local planar guidance for monocular depth estimation
            <https://arxiv.org/abs/1907.10326>`_.

        Example:

        .. code-block:: python

            from torch_uncertainty.metrics.regression import MeanGTRelativeAbsoluteError
            import torch

            # Initialize the metric
            mae_rel_metric = MeanGTRelativeAbsoluteError()

            # Example predictions and targets
            preds = torch.tensor([2.5, 1.0, 2.0, 8.0])
            target = torch.tensor([3.0, 1.5, 2.0, 7.0])

            # Update the metric state
            mae_rel_metric.update(preds, target)

            # Compute the Relative Mean Absolute Error
            result = mae_rel_metric.compute()
            print(f"Relative Mean Absolute Error: {result.item()}")
            # Output: 0.1607142984867096

        .. seealso::
            - :class:`MeanGTRelativeSquaredError`
        """
        super().__init__(**kwargs)

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        return super().update(pred / target, torch.ones_like(target))


class MeanGTRelativeSquaredError(MeanSquaredError):
    def __init__(self, squared: bool = True, num_outputs: int = 1, **kwargs) -> None:
        r"""Compute mean squared error relative to the Ground Truth (MSErel or SRE).

        This metric is useful for evaluating the relative squared error between
        predictions and targets, particularly in regression tasks where relative
        accuracy is critical.

        .. math:: \text{MSErel} = \frac{1}{N}\sum_i^N \frac{(y_i - \hat{y_i})^2}{y_i}

        Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
        tensor of predictions.

        As input to ``forward`` and ``update`` the metric accepts the following
        input:

        - **preds** (:class:`~torch.Tensor`): Predictions from model
        - **target** (:class:`~torch.Tensor`): Ground truth values

        As output of ``forward`` and ``compute`` the metric returns the
        following output:

        - **rel_mean_squared_error** (:class:`~torch.Tensor`): A tensor with
          the relative mean squared error

        Args:
            squared: If True returns MSErel value, if False returns RMSErel value.
            num_outputs: Number of outputs in multioutput setting
            kwargs: Additional keyword arguments, see `Advanced metric settings
              <https://torchmetrics.readthedocs.io/en/stable/pages/overview.html#metric-kwargs>`_.

        Reference:
            [1] `From big to small: Multi-scale local planar guidance for monocular depth estimation
            <https://arxiv.org/abs/1907.10326>`_.

        Example:

        .. code-block:: python

            from torch_uncertainty.metrics.regression import MeanGTRelativeSquaredError
            import torch

            # Initialize the metric
            mse_rel_metric = MeanGTRelativeSquaredError(squared=True)

            # Example predictions and targets
            preds = torch.tensor([2.5, 1.0, 2.0, 8.0])
            target = torch.tensor([3.0, 1.5, 2.0, 7.0])

            # Update the metric state
            mse_rel_metric.update(preds, target)

            # Compute the Relative Mean Squared Error
            result = mse_rel_metric.compute()
            print(f"Relative Mean Squared Error: {result.item()}")
            # Output: 0.09821434319019318

        .. seealso::
            - :class:`MeanGTRelativeAbsoluteError`
        """
        super().__init__(squared, num_outputs, **kwargs)

    def update(self, pred: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        return super().update(pred / torch.sqrt(target), torch.sqrt(target))
