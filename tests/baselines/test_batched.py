import torch
from torch import nn

from torch_uncertainty.baselines.classification import (
    ResNetBaseline,
    WideResNetBaseline,
)


class TestBatchedBaseline:
    """Testing the BatchedResNet baseline class."""

    def test_batched_18(self):
        net = ResNetBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss(),
            version="batched",
            arch=18,
            style="cifar",
            num_estimators=4,
            groups=1,
        )

        _ = net(torch.rand(1, 3, 32, 32))

    def test_batched_50(self):
        net = ResNetBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss(),
            version="batched",
            arch=50,
            style="imagenet",
            num_estimators=4,
            groups=1,
        )

        _ = net(torch.rand(1, 3, 40, 40))


class TestBatchedWideBaseline:
    """Testing the BatchedWideResNet baseline class."""

    def test_batched(self):
        net = WideResNetBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss(),
            version="batched",
            style="cifar",
            num_estimators=4,
            groups=1,
        )

        _ = net(torch.rand(1, 3, 32, 32))
