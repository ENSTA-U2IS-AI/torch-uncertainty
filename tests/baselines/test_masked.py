# fmt:off

from argparse import ArgumentParser

import torch.nn as nn
from torchinfo import summary

from torch_uncertainty.baselines.masked import MaskedResNet
from torch_uncertainty.optimization_procedures import optim_cifar100_resnet18

# fmt:on


class TestMaskedBaseline:
    """Testing the MaskedResNet baseline class."""

    def test_masked(self):
        net = MaskedResNet(
            num_classes=10,
            num_estimators=4,
            in_channels=3,
            scale=2,
            groups=1,
            arch=18,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar100_resnet18,
        )
        parser = ArgumentParser("torch-uncertainty-test")
        parser = net.add_model_specific_args(parser)
        parser.parse_args("")
        summary(net)
