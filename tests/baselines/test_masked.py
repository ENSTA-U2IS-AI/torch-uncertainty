# fmt:off
from argparse import ArgumentParser

import pytest
import torch
import torch.nn as nn
from torchinfo import summary

from torch_uncertainty.baselines.masked import MaskedResNet, MaskedWideResNet
from torch_uncertainty.optimization_procedures import (
    optim_cifar10_wideresnet,
    optim_cifar100_resnet18,
)


# fmt:on
class TestMaskedBaseline:
    """Testing the MaskedResNet baseline class."""

    def test_masked(self):
        net = MaskedResNet(
            arch=18,
            in_channels=3,
            num_classes=10,
            num_estimators=4,
            scale=2,
            groups=1,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar100_resnet18,
            imagenet_structure=False,
        )
        parser = ArgumentParser("torch-uncertainty-test")
        parser = net.add_model_specific_args(parser)
        parser.parse_args(["--no-imagenet_structure"])
        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 32, 32))

    def test_masked_scale_lt_1(self):
        with pytest.raises(Exception):
            _ = MaskedResNet(
                arch=18,
                in_channels=3,
                num_classes=10,
                num_estimators=4,
                scale=0.5,
                groups=1,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar100_resnet18,
            )

    def test_masked_groups_lt_1(self):
        with pytest.raises(Exception):
            _ = MaskedResNet(
                arch=18,
                in_channels=3,
                num_classes=10,
                num_estimators=4,
                scale=2,
                groups=0,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar100_resnet18,
            )


class TestMaskedWideBaseline:
    """Testing the MaskedWideResNet baseline class."""

    def test_masked(self):
        net = MaskedWideResNet(
            arch=18,
            in_channels=3,
            num_classes=10,
            num_estimators=4,
            groups=1,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_wideresnet,
            imagenet_structure=False,
        )
        parser = ArgumentParser("torch-uncertainty-test")
        parser = net.add_model_specific_args(parser)
        parser.parse_args(["--no-imagenet_structure"])
        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 32, 32))
