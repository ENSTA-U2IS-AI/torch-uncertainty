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
        with_linear: bool,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate

        if with_linear:
            self.linear = nn.Linear(
                1,
                num_classes,
            )
        else:
            self.out = nn.Linear(
                1,
                num_classes,
            )
        self.dropout = nn.Dropout(p=dropout_rate)

        self.num_estimators = num_estimators

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(
            self.linear(
                torch.ones(
                    (x.shape[0] * self.num_estimators, 1), dtype=torch.float32
                )
            )
        )


class _DummyWithFeats(_Dummy):
    def feats_forward(self, x: Tensor) -> Tensor:
        return self.forward(x)


def dummy_model(
    in_channels: int,
    num_classes: int,
    num_estimators: int,
    dropout_rate: float = 0.0,
    with_feats: bool = True,
    with_linear: bool = True,
) -> _Dummy:
    """Dummy model for testing purposes.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        num_estimators (int): Number of estimators in the ensemble.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
        with_feats (bool, optional): Whether to include features. Defaults to True.
        with_linear (bool, optional): Whether to include a linear layer.
            Defaults to True.

    Returns:
        _Dummy: Dummy model.
    """
    if with_feats:
        return _DummyWithFeats(
            in_channels=in_channels,
            num_classes=num_classes,
            num_estimators=num_estimators,
            dropout_rate=dropout_rate,
            with_linear=with_linear,
        )
    return _Dummy(
        in_channels=in_channels,
        num_classes=num_classes,
        num_estimators=num_estimators,
        dropout_rate=dropout_rate,
        with_linear=with_linear,
    )
