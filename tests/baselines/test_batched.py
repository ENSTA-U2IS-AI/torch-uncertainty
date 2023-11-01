import torch
from torch import nn
from torchinfo import summary

from torch_uncertainty.baselines import ResNet, WideResNet
from torch_uncertainty.optimization_procedures import (
    optim_cifar10_wideresnet,
    optim_cifar100_resnet18,
    optim_cifar100_resnet50,
)


class TestBatchedBaseline:
    """Testing the BatchedResNet baseline class."""

    def test_batched_18(self):
        net = ResNet(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar100_resnet18,
            version="batched",
            arch=18,
            style="cifar",
            num_estimators=4,
            groups=1,
        )

        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 32, 32))

    def test_batched_50(self):
        net = ResNet(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar100_resnet50,
            version="batched",
            arch=50,
            style="imagenet",
            num_estimators=4,
            groups=1,
        )

        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 40, 40))


class TestBatchedWideBaseline:
    """Testing the BatchedWideResNet baseline class."""

    def test_batched(self):
        net = WideResNet(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_wideresnet,
            version="batched",
            style="cifar",
            num_estimators=4,
            groups=1,
        )

        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 32, 32))
