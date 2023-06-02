# fmt: off

import torch
import torch.nn as nn
from torch import Tensor

# fmt: on
__all__ = [
    "dummy_model",
]


class _Dummy(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_estimators: int,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(
            1,
            num_classes,
        )

        self.num_estimators = num_estimators

    def forward(self, x: Tensor) -> Tensor:
        out = self.linear(
            torch.as_tensor(
                torch.ones(
                    (x.shape[0] * self.num_estimators, 1), dtype=torch.float32
                )
            )
        )
        return out


def dummy_model(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
) -> _Dummy:
    """Dummy model for testing purposes.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        num_estimators (int): Number of estimators in the ensemble.

    Returns:
        _Dummy: Dummy model.
    """
    return _Dummy(
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
    )
