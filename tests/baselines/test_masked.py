import pytest
import torch
from torch import nn
from torchinfo import summary

from torch_uncertainty.baselines import ResNet, WideResNet
from torch_uncertainty.optimization_procedures import (
    optim_cifar10_wideresnet,
    optim_cifar100_resnet18,
    optim_cifar100_resnet50,
)


class TestMaskedBaseline:
    """Testing the MaskedResNet baseline class."""

    def test_masked_18(self):
        net = ResNet(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar100_resnet18,
            version="masked",
            arch=18,
            style="cifar",
            num_estimators=4,
            scale=2,
            groups=1,
        )

        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 32, 32))

    def test_masked_50(self):
        net = ResNet(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar100_resnet50,
            version="masked",
            arch=50,
            style="imagenet",
            num_estimators=4,
            scale=2,
            groups=1,
        )

        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 40, 40))

    def test_masked_scale_lt_1(self):
        with pytest.raises(Exception):
            _ = ResNet(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar100_resnet18,
                version="masked",
                arch=18,
                style="cifar",
                num_estimators=4,
                scale=0.5,
                groups=1,
            )

    def test_masked_groups_lt_1(self):
        with pytest.raises(Exception):
            _ = ResNet(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar100_resnet18,
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
        net = WideResNet(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_wideresnet,
            version="masked",
            style="cifar",
            num_estimators=4,
            scale=2,
            groups=1,
        )

        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 32, 32))
