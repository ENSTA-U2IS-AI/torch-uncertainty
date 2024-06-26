import copy

from torch import Tensor, nn


class EMA(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        momentum: float,
    ) -> None:
        """Exponential Moving Average.

        Args:
            model (nn.Module): The model to train and ensemble.
            momentum (float): The momentum of the moving average.
        """
        super().__init__()
        _ema_checks(momentum)
        self.core_model = model
        self.ema_model = copy.deepcopy(model)
        self.momentum = momentum
        self.remainder = 1 - momentum

    def update_wrapper(self, epoch: int | None = None) -> None:
        """Update the EMA model.

        Args:
            epoch (int): The current epoch. For API consistency.
        """
        for ema_param, param in zip(
            self.ema_model.parameters(),
            self.core_model.parameters(),
            strict=False,
        ):
            ema_param.data = (
                ema_param.data * self.momentum + param.data * self.remainder
            )

    def eval_forward(self, x: Tensor) -> Tensor:
        return self.ema_model.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return self.core_model.forward(x)
        return self.eval_forward(x)


def _ema_checks(momentum: float) -> None:
    if momentum < 0.0 or momentum >= 1.0:
        raise ValueError(
            f"`momentum` must be in the range [0, 1). Got {momentum}."
        )
