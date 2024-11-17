import pytest
import torch
from torch import nn

from torch_uncertainty.baselines.classification import (
    ResNetBaseline,
    VGGBaseline,
    WideResNetBaseline,
)
from torch_uncertainty.baselines.regression import MLPBaseline


class TestPackedBaseline:
    """Testing the PackedResNet baseline class."""

    def test_packed_50(self):
        net = ResNetBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss(),
            version="packed",
            arch=50,
            style="cifar",
            num_estimators=4,
            alpha=2,
            gamma=1,
            groups=1,
        )

        _ = net(torch.rand(1, 3, 32, 32))

    def test_packed_18(self):
        net = ResNetBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss(),
            version="packed",
            arch=18,
            style="imagenet",
            num_estimators=4,
            alpha=2,
            gamma=2,
            groups=2,
        )

        _ = net(torch.rand(1, 3, 40, 40))

    def test_packed_exception(self):
        with pytest.raises(ValueError):
            _ = ResNetBaseline(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss(),
                version="packed",
                arch=50,
                style="cifar",
                num_estimators=4,
                alpha=0,
                gamma=1,
                groups=1,
            )

        with pytest.raises(ValueError):
            _ = ResNetBaseline(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss(),
                version="packed",
                arch=50,
                style="cifar",
                num_estimators=4,
                alpha=2,
                gamma=0,
                groups=1,
            )


class TestPackedWideBaseline:
    """Testing the PackedWideResNet baseline class."""

    def test_packed(self):
        net = WideResNetBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss(),
            version="packed",
            style="cifar",
            num_estimators=4,
            alpha=2,
            gamma=1,
            groups=1,
        )

        _ = net(torch.rand(1, 3, 32, 32))


class TestPackedVGGBaseline:
    """Testing the PackedWideResNet baseline class."""

    def test_packed(self):
        net = VGGBaseline(
            num_classes=10,
            in_channels=3,
            arch=13,
            loss=nn.CrossEntropyLoss(),
            version="packed",
            num_estimators=4,
            alpha=2,
            gamma=1,
            groups=1,
        )
        _ = net(torch.rand(2, 3, 32, 32))


class TestPackedMLPBaseline:
    """Testing the Packed MLP baseline class."""

    def test_packed(self):
        net = MLPBaseline(
            in_features=3,
            output_dim=10,
            loss=nn.MSELoss(),
            version="packed",
            hidden_dims=[1],
            num_estimators=2,
            alpha=2,
            gamma=1,
        )
        _ = net(torch.rand(1, 3))
