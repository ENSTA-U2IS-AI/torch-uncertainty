from typing import Literal

from torch import Tensor
from torchmetrics import MeanAbsoluteError, MeanSquaredError


def _unit_to_factor(unit: Literal["mm", "m", "km"]) -> float:
    """Convert a unit to a factor for scaling.

    Args:
        unit: Unit for the computation of the metric. Must be one of 'mm', 'm',
            'km'.
    """
    if unit == "km":
        return 1e-3
    if unit == "m":
        return 1.0
    if unit == "mm":
        return 1e3
    raise ValueError(f"unit must be one of 'mm', 'm', 'km'. Got {unit}.")


class MeanSquaredErrorInverse(MeanSquaredError):
    r"""Mean Squared Error of the inverse predictions (iMSE).

    .. math:: \text{iMSE} = \frac{1}{N}\sum_i^N(\frac{1}{y_i} - \frac{1}{\hat{y_i}})^2

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
        tensor of predictions.
    Both are scaled by a factor of :attr:`unit_factor` depending on the
        :attr:`unit` given.

    As input to ``forward`` and ``update`` the metric accepts the following
        input:

    - ``preds`` (:class:`~Tensor`): Predictions from model
    - ``target`` (:class:`~Tensor`): Ground truth values

    As output of ``forward`` and ``compute`` the metric returns the following
        output:

    - ``mean_squared_error`` (:class:`~Tensor`): A tensor with the mean
        squared error

    Args:
        squared: If True returns MSE value, if False returns RMSE value.
        num_outputs: Number of outputs in multioutput setting.
        unit: Unit for the computation of the metric. Must be one of 'mm', 'm',
            'km'. Defauts to 'km'.
        kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        squared: bool = True,
        num_outputs: int = 1,
        unit: str = "km",
        **kwargs,
    ) -> None:
        super().__init__(squared, num_outputs, **kwargs)
        self.unit_factor = _unit_to_factor(unit)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        super().update(
            1 / (preds * self.unit_factor), 1 / (target * self.unit_factor)
        )


class MeanAbsoluteErrorInverse(MeanAbsoluteError):
    r"""Mean Absolute Error of the inverse predictions (iMAE).

    .. math:: \text{iMAE} = \frac{1}{N}\sum_i^N | \frac{1}{y_i} - \frac{1}{\hat{y_i}} |

    Where :math:`y` is a tensor of target values, and :math:`\hat{y}` is a
        tensor of predictions.
    Both are scaled by a factor of :attr:`unit_factor` depending on the
        :attr:`unit` given.

    As input to ``forward`` and ``update`` the metric accepts the following
        input:

    - ``preds`` (:class:`~Tensor`): Predictions from model
    - ``target`` (:class:`~Tensor`): Ground truth values

    As output of ``forward`` and ``compute`` the metric returns the following
        output:

    - ``mean_absolute_inverse_error`` (:class:`~Tensor`): A tensor with the
        mean absolute error over the state

    Args:
        unit: Unit for the computation of the metric. Must be one of 'mm', 'm',
            'km'. Defauts to 'km'.
        kwargs: Additional keyword arguments.
    """

    def __init__(self, unit: str = "km", **kwargs) -> None:
        super().__init__(**kwargs)
        self.unit_factor = _unit_to_factor(unit)

    def update(self, preds: Tensor, target: Tensor) -> None:
        """Update state with predictions and targets."""
        super().update(
            1 / (preds * self.unit_factor), 1 / (target * self.unit_factor)
        )
