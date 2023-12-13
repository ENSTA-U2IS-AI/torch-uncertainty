import torch
from torch import Tensor, nn

__all__ = [
    "dummy_model",
]


class _Dummy(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        num_estimators: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate

        self.linear = nn.Linear(
            1,
            num_classes,
        )

        self.num_estimators = num_estimators

    def feats_forward(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(
            torch.ones(
                (x.shape[0] * self.num_estimators, 1), dtype=torch.float32
            )
        )


def dummy_model(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    dropout_rate: float = 0.0,
) -> _Dummy:
    """Dummy model for testing purposes.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        num_estimators (int): Number of estimators in the ensemble.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.0.

    Returns:
        _Dummy: Dummy model.
    """
    return _Dummy(
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        dropout_rate=dropout_rate,
    )
