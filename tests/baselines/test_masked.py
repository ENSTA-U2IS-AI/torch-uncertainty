import pytest
import torch
from torch import nn

from torch_uncertainty.baselines.classification import (
    ResNetBaseline,
    WideResNetBaseline,
)


class TestMaskedBaseline:
    """Testing the MaskedResNet baseline class."""

    def test_masked_18(self):
        net = ResNetBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss(),
            version="masked",
            arch=18,
            style="cifar",
            num_estimators=4,
            scale=2,
            groups=1,
        )

        _ = net(torch.rand(1, 3, 32, 32))

    def test_masked_50(self):
        net = ResNetBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss(),
            version="masked",
            arch=50,
            style="imagenet",
            num_estimators=4,
            scale=2,
            groups=1,
        )

        _ = net(torch.rand(1, 3, 40, 40))

    def test_masked_errors(self):
        with pytest.raises(ValueError):
            _ = ResNetBaseline(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss(),
                version="masked",
                arch=18,
                style="cifar",
                num_estimators=4,
                scale=0.5,
                groups=1,
            )

        with pytest.raises(ValueError):
            _ = ResNetBaseline(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss(),
                version="masked",
                arch=18,
                style="cifar",
                num_estimators=4,
                scale=2,
                groups=0,
            )


class TestMaskedWideBaseline:
    """Testing the MaskedWideResNet baseline class."""

    def test_masked(self):
        net = WideResNetBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss(),
            version="masked",
            style="cifar",
            num_estimators=4,
            scale=2,
            groups=1,
        )

        _ = net(torch.rand(1, 3, 32, 32))
