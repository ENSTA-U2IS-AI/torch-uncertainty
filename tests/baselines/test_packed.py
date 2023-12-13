import pytest
import torch
from torch import nn
from torchinfo import summary

from torch_uncertainty.baselines import VGG, ResNet, WideResNet
from torch_uncertainty.baselines.regression import MLP
from torch_uncertainty.optimization_procedures import (
    optim_cifar10_resnet18,
    optim_cifar10_resnet50,
    optim_cifar10_wideresnet,
)


class TestPackedBaseline:
    """Testing the PackedResNet baseline class."""

    def test_packed_50(self):
        net = ResNet(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_resnet50,
            version="packed",
            arch=50,
            style="cifar",
            num_estimators=4,
            alpha=2,
            gamma=1,
            groups=1,
        )

        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 32, 32))

    def test_packed_18(self):
        net = ResNet(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_resnet18,
            version="packed",
            arch=18,
            style="imagenet",
            num_estimators=4,
            alpha=2,
            gamma=2,
            groups=2,
        )

        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 40, 40))

    def test_packed_exception(self):
        with pytest.raises(Exception):
            _ = ResNet(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet50,
                version="packed",
                arch=50,
                style="cifar",
                num_estimators=4,
                alpha=0,
                gamma=1,
                groups=1,
            )

        with pytest.raises(Exception):
            _ = ResNet(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet50,
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
        net = WideResNet(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_wideresnet,
            version="packed",
            style="cifar",
            num_estimators=4,
            alpha=2,
            gamma=1,
            groups=1,
        )

        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 32, 32))


class TestPackedVGGBaseline:
    """Testing the PackedWideResNet baseline class."""

    def test_packed(self):
        net = VGG(
            num_classes=10,
            in_channels=3,
            arch=13,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_resnet50,
            version="packed",
            num_estimators=4,
            alpha=2,
            gamma=1,
            groups=1,
        )

        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(2, 3, 32, 32))


class TestPackedMLPBaseline:
    """Testing the Packed MLP baseline class."""

    def test_packed(self):
        net = MLP(
            in_features=3,
            num_outputs=10,
            loss=nn.MSELoss,
            optimization_procedure=optim_cifar10_resnet18,
            version="packed",
            hidden_dims=[1],
            num_estimators=2,
            alpha=2,
            gamma=1,
            dist_estimation=1,
        )
        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3))
