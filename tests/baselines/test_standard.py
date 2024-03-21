import pytest
import torch
from torch import nn
from torchinfo import summary

from torch_uncertainty.baselines.classification import (
    ResNetBaseline,
    VGGBaseline,
    WideResNetBaseline,
)
from torch_uncertainty.baselines.regression import MLP
from torch_uncertainty.baselines.segmentation import SegFormer


class TestStandardBaseline:
    """Testing the ResNetBaseline baseline class."""

    def test_standard(self):
        net = ResNetBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            version="std",
            arch=18,
            style="cifar",
            groups=1,
        )
        summary(net)

        _ = net.criterion
        _ = net(torch.rand(1, 3, 32, 32))

    def test_errors(self):
        with pytest.raises(ValueError):
            ResNetBaseline(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss,
                version="test",
                arch=18,
                style="cifar",
                groups=1,
            )


class TestStandardWideBaseline:
    """Testing the WideResNetBaseline baseline class."""

    def test_standard(self):
        net = WideResNetBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            version="std",
            style="cifar",
            groups=1,
        )
        summary(net)

        _ = net.criterion
        _ = net(torch.rand(1, 3, 32, 32))

    def test_errors(self):
        with pytest.raises(ValueError):
            WideResNetBaseline(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss,
                version="test",
                style="cifar",
                groups=1,
            )


class TestStandardVGGBaseline:
    """Testing the VGGBaseline baseline class."""

    def test_standard(self):
        net = VGGBaseline(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            version="std",
            arch=11,
            groups=1,
        )
        summary(net)

        _ = net.criterion
        _ = net(torch.rand(1, 3, 32, 32))

    def test_errors(self):
        with pytest.raises(ValueError):
            VGGBaseline(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss,
                version="test",
                arch=11,
                groups=1,
            )


class TestStandardMLPBaseline:
    """Testing the MLP baseline class."""

    def test_standard(self):
        net = MLP(
            in_features=3,
            num_outputs=10,
            loss=nn.MSELoss,
            version="std",
            hidden_dims=[1],
        )
        summary(net)

        _ = net.criterion
        _ = net(torch.rand(1, 3))

    def test_errors(self):
        with pytest.raises(ValueError):
            MLP(
                in_features=3,
                num_outputs=10,
                loss=nn.MSELoss,
                version="test",
                hidden_dims=[1],
            )


class TestStandardSegFormerBaseline:
    """Testing the SegFormer baseline class."""

    def test_standard(self):
        net = SegFormer(
            num_classes=10,
            loss=nn.CrossEntropyLoss,
            version="std",
            arch=0,
        )
        summary(net)

        _ = net.criterion
        _ = net(torch.rand(1, 3, 32, 32))

    def test_errors(self):
        with pytest.raises(ValueError):
            SegFormer(
                num_classes=10,
                loss=nn.CrossEntropyLoss,
                version="test",
                arch=0,
            )
