# fmt:off

import torch.nn as nn
from torchinfo import summary

from torch_uncertainty.baselines.packed import PackedResNet
from torch_uncertainty.optimization_procedures import optim_cifar10_resnet50

# fmt:on


class TestPackedBaseline:
    """Testing the PackedResNet baseline class."""

    def test_packed(self):
        net = PackedResNet(
            10, 4, 3, 2, 1, 50, nn.CrossEntropyLoss, optim_cifar10_resnet50
        )
        summary(net)
