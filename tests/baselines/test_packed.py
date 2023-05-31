# fmt:off
import pytest
import torch
import torch.nn as nn
from torchinfo import summary

from torch_uncertainty.baselines import ResNet, WideResNet
from torch_uncertainty.optimization_procedures import (
    optim_cifar10_resnet50,
    optim_cifar10_wideresnet,
)


# fmt:on
class TestPackedBaseline:
    """Testing the PackedResNet baseline class."""

    def test_packed(self):
        net = ResNet(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_resnet50,
            version="packed",
            arch=50,
            imagenet_structure=False,
            num_estimators=4,
            alpha=2,
            gamma=1,
            groups=1,
        )

        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 32, 32))

    def test_packed_alpha_lt_0(self):
        with pytest.raises(Exception):
            _ = ResNet(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet50,
                version="packed",
                arch=50,
                imagenet_structure=False,
                num_estimators=4,
                alpha=0,
                gamma=1,
                groups=1,
            )

    def test_packed_gamma_lt_1(self):
        with pytest.raises(Exception):
            _ = ResNet(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet50,
                version="packed",
                arch=50,
                imagenet_structure=False,
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
            imagenet_structure=False,
            num_estimators=4,
            alpha=2,
            gamma=1,
            groups=1,
        )

        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 32, 32))
