# fmt:off
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torchinfo import summary

from torch_uncertainty.baselines.batched import BatchedResNet, BatchedWideResNet
from torch_uncertainty.optimization_procedures import (
    optim_cifar10_wideresnet,
    optim_cifar100_resnet50,
)


# fmt:on
class TestBatchedBaseline:
    """Testing the BatchedResNet baseline class."""

    def test_batched(self):
        net = BatchedResNet(
            arch=18,
            in_channels=3,
            num_classes=10,
            num_estimators=4,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar100_resnet50,
            imagenet_structure=False,
        )
        parser = ArgumentParser("torch-uncertainty-test")
        parser = net.add_model_specific_args(parser)
        parser.parse_args(["--no-imagenet_structure"])
        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 32, 32))


class TestBatchedWideBaseline:
    """Testing the BatchedWideResNet baseline class."""

    def test_batched(self):
        net = BatchedWideResNet(
            arch=18,
            num_classes=10,
            num_estimators=4,
            in_channels=3,
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
