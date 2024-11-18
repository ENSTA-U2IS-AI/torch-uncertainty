import torch
from torch import nn

from torch_uncertainty.baselines.classification import (
    ResNetBaseline,
    WideResNetBaseline,
)


class TestMIMOBaseline:
    """Testing the MIMOResNet baseline class."""

    def test_mimo_50(self):
        net = ResNetBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss(),
            version="mimo",
            arch=50,
            style="cifar",
            num_estimators=4,
            rho=0.5,
            batch_repeat=4,
            groups=1,
        ).eval()

        _ = net(torch.rand(1, 3, 32, 32))

    def test_mimo_18(self):
        net = ResNetBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss(),
            version="mimo",
            arch=18,
            style="imagenet",
            num_estimators=4,
            rho=0.5,
            batch_repeat=4,
            groups=2,
        ).eval()

        _ = net(torch.rand(1, 3, 40, 40))


class TestMIMOWideBaseline:
    """Testing the PackedWideResNet baseline class."""

    def test_mimo(self):
        net = WideResNetBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss(),
            version="mimo",
            style="cifar",
            num_estimators=4,
            rho=0.5,
            batch_repeat=4,
            groups=1,
        ).eval()

        _ = net(torch.rand(1, 3, 32, 32))
