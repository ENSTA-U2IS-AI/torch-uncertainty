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
            num_classes=10,
            num_estimators=4,
            in_channels=3,
            arch=18,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_cifar100_resnet50,
        )
        summary(net)
