# fmt:off
import pytest
import torch.nn as nn

from torch_uncertainty.baselines import ResNet, WideResNet
from torch_uncertainty.optimization_procedures import (
    optim_cifar10_resnet50,
    optim_cifar10_wideresnet,
)


# fmt:on
class TestResnetBaseline:
    """Testing the ResNet baseline class."""

    def test_resnet_unknown_version(self):
        with pytest.raises(ValueError):
            _ = ResNet(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet50,
                arch=50,
                version="other",
            )


class TestWideResnetBaseline:
    """Testing the ResNet baseline class."""

    def test_resnet_unknown_version(self):
        with pytest.raises(ValueError):
            _ = WideResNet(
                num_classes=10,
                in_channels=3,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_wideresnet,
                version="other",
            )
