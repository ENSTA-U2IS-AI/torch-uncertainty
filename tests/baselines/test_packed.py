# fmt:off
from argparse import ArgumentParser

import pytest
import torch
import torch.nn as nn
from torchinfo import summary

from torch_uncertainty.baselines.packed import PackedResNet, PackedWideResNet
from torch_uncertainty.optimization_procedures import (
    optim_cifar10_resnet50,
    optim_cifar10_wideresnet,
)


# fmt:on
class TestPackedBaseline:
    """Testing the PackedResNet baseline class."""

    def test_packed(self):
        net = PackedResNet(
            num_classes=10,
            num_estimators=4,
            in_channels=3,
            alpha=2,
            gamma=1,
            arch=50,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_resnet50,
            imagenet_structure=False,
        )
        parser = ArgumentParser("torch-uncertainty-test")
        parser = net.add_model_specific_args(parser)
        parser.parse_args(["--arch", "50", "--no-imagenet_structure"])
        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 32, 32))

    def test_packed_alpha_lt_0(self):
        with pytest.raises(Exception):
            _ = PackedResNet(
                num_classes=10,
                num_estimators=4,
                in_channels=3,
                alpha=0,
                gamma=1,
                arch=50,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet50,
            )

    def test_packed_gamma_lt_1(self):
        with pytest.raises(Exception):
            _ = PackedResNet(
                num_classes=10,
                num_estimators=4,
                in_channels=3,
                alpha=2,
                gamma=0,
                arch=50,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet50,
            )


class TestPackedWideBaseline:
    """Testing the PackedWideResNet baseline class."""

    def test_packed(self):
        net = PackedWideResNet(
            num_classes=10,
            num_estimators=4,
            in_channels=3,
            alpha=2,
            gamma=1,
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
