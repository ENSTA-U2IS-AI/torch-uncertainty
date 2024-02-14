from argparse import ArgumentParser

import pytest
import torch
from torch import nn
from torchinfo import summary

from torch_uncertainty.baselines import VGG, ResNet, WideResNet
from torch_uncertainty.baselines.regression import MLP
from torch_uncertainty.baselines.utils.parser_addons import (
    add_mlp_specific_args,
)
from torch_uncertainty.optimization_procedures import (
    optim_cifar10_resnet18,
    optim_cifar10_wideresnet,
)


class TestStandardBaseline:
    """Testing the ResNet baseline class."""

    def test_standard(self):
        net = ResNet(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_resnet18,
            version="std",
            arch=18,
            style="cifar",
            groups=1,
        )
        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 32, 32))

    def test_errors(self):
        with pytest.raises(ValueError):
            ResNet(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet18,
                version="test",
                arch=18,
                style="cifar",
                groups=1,
            )


class TestStandardWideBaseline:
    """Testing the WideResNet baseline class."""

    def test_standard(self):
        net = WideResNet(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_wideresnet,
            version="std",
            style="cifar",
            groups=1,
        )
        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 32, 32))

    def test_errors(self):
        with pytest.raises(ValueError):
            WideResNet(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_wideresnet,
                version="test",
                style="cifar",
                groups=1,
            )


class TestStandardVGGBaseline:
    """Testing the VGG baseline class."""

    def test_standard(self):
        net = VGG(
            num_classes=10,
            in_channels=3,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_resnet18,
            version="std",
            arch=11,
            groups=1,
        )
        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 32, 32))

    def test_errors(self):
        with pytest.raises(ValueError):
            VGG(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet18,
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
            optimization_procedure=optim_cifar10_resnet18,
            version="std",
            hidden_dims=[1],
            dist_estimation=1,
        )
        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3))

        parser = ArgumentParser()
        add_mlp_specific_args(parser)

    def test_errors(self):
        with pytest.raises(ValueError):
            MLP(
                in_features=3,
                num_outputs=10,
                loss=nn.MSELoss,
                optimization_procedure=optim_cifar10_resnet18,
                version="test",
                hidden_dims=[1],
                dist_estimation=1,
            )
