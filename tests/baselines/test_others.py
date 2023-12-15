import pytest
from torch import nn

from torch_uncertainty.baselines import VGG, ResNet, WideResNet
from torch_uncertainty.baselines.regression import MLP
from torch_uncertainty.optimization_procedures import (
    optim_cifar10_resnet18,
    optim_cifar10_wideresnet,
)


class TestStandardBaseline:
    """Testing the ResNet baseline class."""

    def test_standard(self):
        with pytest.raises(ValueError):
            ResNet(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet18,
                version="prior",
                arch=18,
                style="cifar",
                groups=1,
            )


class TestStandardWideBaseline:
    """Testing the WideResNet baseline class."""

    def test_standard(self):
        with pytest.raises(ValueError):
            WideResNet(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_wideresnet,
                version="prior",
                style="cifar",
                groups=1,
            )


class TestStandardVGGBaseline:
    """Testing the VGG baseline class."""

    def test_standard(self):
        with pytest.raises(ValueError):
            VGG(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet18,
                version="prior",
                arch=11,
                groups=1,
            )


class TestStandardMLPBaseline:
    """Testing the MLP baseline class."""

    def test_standard(self):
        with pytest.raises(ValueError):
            MLP(
                in_features=3,
                num_outputs=10,
                loss=nn.MSELoss,
                optimization_procedure=optim_cifar10_resnet18,
                version="prior",
                hidden_dims=[1],
                dist_estimation=1,
            )
