# fmt:off

from argparse import ArgumentParser

import torch
import torch.nn as nn
from torchinfo import summary

from torch_uncertainty.baselines.standard import ResNet
from torch_uncertainty.optimization_procedures import optim_cifar10_resnet18


# fmt:on
class TestStandardBaseline:
    """Testing the ResNet baseline class."""

    def test_packed(self):
        net = ResNet(
            num_classes=10,
            in_channels=3,
            groups=1,
            arch=34,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar10_resnet18,
            imagenet_structure=False,
        )
        parser = ArgumentParser("torch-uncertainty-test")
        parser = net.add_model_specific_args(parser)
        parser.parse_args(["--no-imagenet_structure"])
        summary(net)

        _ = net.criterion
        _ = net.configure_optimizers()
        _ = net(torch.rand(1, 3, 32, 32))
