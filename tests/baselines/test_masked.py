# fmt:off

import torch.nn as nn
from torchinfo import summary

from torch_uncertainty.baselines.masked import MaskedResNet
from torch_uncertainty.optimization_procedures import optim_cifar100_resnet18

# fmt:on


class TestMaskedBaseline:
    """Testing the MaskedResNet baseline class."""

    def test_masked(self):
        net = MaskedResNet(
            10, 4, 3, 2, 1, 18, nn.CrossEntropyLoss, optim_cifar100_resnet18
        )
        summary(net)
