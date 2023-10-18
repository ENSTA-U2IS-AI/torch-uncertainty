# fmt:off
import pytest
import torch
from torch import nn
from torchinfo import summary

from torch_uncertainty.baselines import ResNet, WideResNet
from torch_uncertainty.optimization_procedures import (
    optim_cifar10_resnet18,
    optim_cifar10_resnet50,
    optim_cifar10_wideresnet,
)


# fmt:on
class TestResnetBaseline:
    """Testing the ResNet baseline class."""

    def test_resnet_unknown_version(self):
        with pytest.raises(ValueError):
            _ = ResNet(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet50,
                arch=50,
                version="other",
            )

    def test_mc_dropout_resnet18(self):
        net = ResNet(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_resnet18,
            version="mc-dropout",
            arch=18,
            style="cifar",
            num_estimators=4,
            groups=1,
        )

        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 32, 32))

    def test_mc_dropout_resnet50(self):
        net = ResNet(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_resnet50,
            version="mc-dropout",
            arch=50,
            style="imagenet",
            num_estimators=4,
            groups=1,
        )

        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 40, 40))


class TestWideResnetBaseline:
    """Testing the ResNet baseline class."""

    def test_resnet_unknown_version(self):
        with pytest.raises(ValueError):
            _ = WideResNet(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_wideresnet,
                version="other",
            )
