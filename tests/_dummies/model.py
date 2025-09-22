import torch
from torch import Tensor, nn

from torch_uncertainty.layers.distributions import get_dist_conv_layer, get_dist_linear_layer

__all__ = [
    "dummy_model",
]


class _Dummy(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout_rate: float,
        dist_family: str | None = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate

        self.linear = nn.Linear(1, num_classes)

        if dist_family is None:
            self.last_layer = nn.Linear(num_classes, num_classes)
        else:
            self.last_layer = get_dist_linear_layer(dist_family)(
                base_layer=nn.Linear,
                event_dim=num_classes,
                in_features=num_classes,
            )
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        return self.last_layer(
            self.dropout(
                self.linear(
                    torch.ones(
                        (x.shape[0], 1),
                        dtype=torch.float32,
                        device=x.device,
                    )
                )
            )
        )


class _DummyWithFeats(_Dummy):
    def feats_forward(self, x: Tensor) -> Tensor:
        return torch.ones(
            (x.shape[0], 1),
            dtype=torch.float32,
            device=x.device,
        )


class _DummySegmentation(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        dropout_rate: float,
        image_size: int,
        dist_family: str | None = None,
    ) -> None:
        super().__init__()
        self.dropout_rate = dropout_rate
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.image_size = image_size
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(p=dropout_rate)
        if dist_family is None:
            self.last_layer = nn.Identity()
        else:
            self.last_layer = get_dist_conv_layer(dist_family)(
                base_layer=nn.Conv2d,
                event_dim=num_classes,
                in_channels=num_classes,
                kernel_size=3,
                padding=1,
            )

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
                        device=x.device,
                    )
                )
            )
        )


def dummy_model(
    in_channels: int,
    num_classes: int,
    dropout_rate: float = 0.0,
    with_feats: bool = True,
    dist_family: str | None = None,
) -> _Dummy:
    """Dummy model for testing purposes.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        num_estimators (int): Number of estimators in the ensemble.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
        with_feats (bool, optional): Whether to include features. Defaults to True.
        dist_family (str, optional): Distribution family. Defaults to None.

    Returns:
        _Dummy: Dummy model.
    """
    if with_feats:
        return _DummyWithFeats(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout_rate=dropout_rate,
            dist_family=dist_family,
        )
    return _Dummy(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        dist_family=dist_family,
    )


def dummy_segmentation_model(
    in_channels: int,
    num_classes: int,
    image_size: int,
    dropout_rate: float = 0.0,
    dist_family: str | None = None,
) -> nn.Module:
    """Dummy segmentation model for testing purposes.

    Args:
        in_channels (int): Number of input channels.
        num_classes (int): Number of output classes.
        image_size (int): Size of the input image.
        dropout_rate (float, optional): Dropout rate. Defaults to 0.0.
        dist_family (str, optional): Distribution family. Defaults to None.

    Returns:
        nn.Module: Dummy segmentation model.
    """
    return _DummySegmentation(
        in_channels=in_channels,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
        image_size=image_size,
        dist_family=dist_family,
    )


class dummy_ood_model(nn.Module):  # noqa: N801
    def __init__(self, in_channels=3, feat_dim=4096, num_classes=3):
        super().__init__()
        self.feat = nn.Sequential(
            nn.Conv2d(in_channels, 8, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, feat_dim),
            nn.ReLU(),
        )
        self.norm = nn.LayerNorm(feat_dim)
        self.fc = nn.Linear(feat_dim, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.02, 0.02)
                nn.init.zeros_(m.bias)
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight, -0.02, 0.02)
                nn.init.zeros_(m.bias)

        self.feature_size = feat_dim

    def forward(self, x, return_feature=False, return_feature_list=False):
        f = self.feat(x)
        f = torch.tanh(self.norm(f))
        logits = self.fc(f)
        if return_feature:
            return logits, f
        return logits

    def get_fc(self):
        w = self.fc.weight.detach().cpu().numpy()
        b = self.fc.bias.detach().cpu().numpy()
        return w, b

    def get_fc_layer(self):
        return self.fc
