import torch
from torch import nn

from torch_uncertainty.baselines.classification import (
    ResNetBaseline,
    VGGBaseline,
    WideResNetBaseline,
)


class TestStandardBaseline:
    """Testing the ResNetBaseline baseline class."""

    def test_standard(self):
        net = ResNetBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss(),
            version="mc-dropout",
            dropout_rate=0.1,
            num_estimators=4,
            arch=18,
            style="cifar",
            groups=1,
        )
        net(torch.rand(1, 3, 32, 32))


class TestStandardWideBaseline:
    """Testing the WideResNetBaseline baseline class."""

    def test_standard(self):
        net = WideResNetBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss(),
            version="mc-dropout",
            dropout_rate=0.1,
            num_estimators=4,
            style="cifar",
            groups=1,
        )
        net(torch.rand(1, 3, 32, 32))


class TestStandardVGGBaseline:
    """Testing the VGGBaseline baseline class."""

    def test_standard(self):
        net = VGGBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss(),
            version="mc-dropout",
            dropout_rate=0.1,
            num_estimators=4,
            arch=11,
            groups=1,
            last_layer_dropout=True,
        )
        net(torch.rand(1, 3, 32, 32))

        net = VGGBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss(),
            version="mc-dropout",
            num_estimators=4,
            arch=11,
            groups=1,
            dropout_rate=0.3,
            last_layer_dropout=True,
        )
        net.eval()
        net(torch.rand(1, 3, 32, 32))
