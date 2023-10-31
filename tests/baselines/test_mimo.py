import torch
from torch import nn
from torchinfo import summary

from torch_uncertainty.baselines import ResNet, WideResNet
from torch_uncertainty.optimization_procedures import (
    optim_cifar10_resnet18,
    optim_cifar10_resnet50,
    optim_cifar10_wideresnet,
)


class TestMIMOBaseline:
    """Testing the MIMOResNet baseline class."""

    def test_mimo_50(self):
        net = ResNet(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_resnet50,
            version="mimo",
            arch=50,
            style="cifar",
            num_estimators=4,
            rho=0.5,
            batch_repeat=4,
            groups=1,
        ).eval()

        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 32, 32))

    def test_mimo_18(self):
        net = ResNet(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_resnet18,
            version="mimo",
            arch=18,
            style="imagenet",
            num_estimators=4,
            rho=0.5,
            batch_repeat=4,
            groups=2,
        ).eval()

        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 40, 40))


class TestMIMOWideBaseline:
    """Testing the PackedWideResNet baseline class."""

    def test_mimo(self):
        net = WideResNet(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_wideresnet,
            version="mimo",
            style="cifar",
            num_estimators=4,
            rho=0.5,
            batch_repeat=4,
            groups=1,
        ).eval()

        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 32, 32))
