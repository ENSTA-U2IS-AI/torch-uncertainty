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
        dropout_rate: float,
        last_layer: nn.Module,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate

        self.linear = nn.Linear(
            1,
            num_classes,
        )
        self.last_layer = last_layer
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        return self.last_layer(
            self.dropout(
                self.linear(
                    torch.ones(
                        (x.shape[0], 1),
                        dtype=torch.float32,
                    )
                )
            )
        )


class _DummyWithFeats(_Dummy):
    def feats_forward(self, x: Tensor) -> Tensor:
        return torch.ones(
            (x.shape[0], 1),
            dtype=torch.float32,
        )


class _DummySegmentation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout_rate: float,
        image_size: int,
        last_layer: nn.Module,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.conv = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, padding=1
        )
        self.dropout = nn.Dropout(p=dropout_rate)
        self.last_layer = last_layer

    def forward(self, x: Tensor) -> Tensor:
        return self.last_layer(
            self.dropout(
                self.conv(
                    torch.ones(
                        (
                            x.shape[0],
                            self.in_channels,
                            self.image_size,
                            self.image_size,
                        ),
                        dtype=torch.float32,
                    )
                )
            )
        )


def dummy_model(
    in_channels: int,
    num_classes: int,
    dropout_rate: float = 0.0,
    with_feats: bool = True,
    last_layer: nn.Module | None = None,
) -> _Dummy:
    """Dummy model for testing purposes.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        num_estimators (int): Number of estimators in the ensemble.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
        with_feats (bool, optional): Whether to include features. Defaults to True.
        last_layer (nn.Module, optional): Last layer of the model. Defaults to None.

    Returns:
        _Dummy: Dummy model.
    """
    if last_layer is None:
        last_layer = nn.Identity()
    if with_feats:
        return _DummyWithFeats(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            last_layer=last_layer,
        )
    return _Dummy(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        last_layer=last_layer,
    )


def dummy_segmentation_model(
    in_channels: int,
    num_classes: int,
    image_size: int,
    dropout_rate: float = 0.0,
    last_layer: nn.Module | None = None,
) -> nn.Module:
    """Dummy segmentation model for testing purposes.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        image_size (int): Size of the input image.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
        last_layer (nn.Module, optional): Last layer of the model. Defaults to None.

    Returns:
        nn.Module: Dummy segmentation model.
    """
    if last_layer is None:
        last_layer = nn.Identity()
    return _DummySegmentation(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        image_size=image_size,
        last_layer=last_layer,
    )
