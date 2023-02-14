# fmt:off

import torch.nn as nn
from torchinfo import summary

from torch_uncertainty.baselines.batched import BatchedResNet
from torch_uncertainty.optimization_procedures import optim_cifar100_resnet50

# fmt:on


class TestBatchedBaseline:
    """Testing the BatchedResNet baseline class."""

    def test_batched(self):
        net = BatchedResNet(
            10, 4, 3, 2, 1, 18, nn.CrossEntropyLoss, optim_cifar100_resnet50
        )
        summary(net)
