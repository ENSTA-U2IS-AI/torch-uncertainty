# flake8: noqa
# fmt: off
import pytest

from torch_uncertainty.models.resnet import resnet18, resnet50
from torch_uncertainty.models.wideresnet import wideresnet28x10
from torch_uncertainty.optimization_procedures import *
from torch_uncertainty.optimization_procedures import get_procedure


# fmt: on
class TestOptProcedures:
    def test_optim_cifar10_resnet18(self):
        procedure = get_procedure("resnet18", "cifar10", "standard")
        model = resnet18(in_channels=3, num_classes=10)
        procedure(model)

    def test_optim_cifar10_resnet50(self):
        procedure = get_procedure("resnet50", "cifar10", "packed")
        model = resnet50(in_channels=3, num_classes=10)
        procedure(model)

    def test_optim_cifar10_wideresnet(self):
        procedure = get_procedure("wideresnet28x10", "cifar100", "batched")
        model = wideresnet28x10(in_channels=3, num_classes=10)
        procedure(model)

    def test_optim_cifar100_resnet18(self):
        procedure = get_procedure("resnet18", "cifar100", "masked")
        model = resnet18(in_channels=3, num_classes=100)
        procedure(model)

    def test_optim_cifar100_resnet50(self):
        procedure = get_procedure("resnet50", "cifar100")
        model = resnet50(in_channels=3, num_classes=100)
        procedure(model)

    def test_optim_cifar100_wideresnet(self):
        procedure = get_procedure("wideresnet28x10", "cifar100")
        model = wideresnet28x10(in_channels=3, num_classes=100)
        procedure(model)

    def test_optim_imagenet_resnet50(self):
        model = resnet50(in_channels=3, num_classes=1000)
        optim_imagenet_resnet50(model)

    def test_optim_imagenet_resnet50_A3(self):
        model = resnet50(in_channels=3, num_classes=1000)
        optim_imagenet_resnet50_A3(model)

    def test_optim_unknown(self):
        with pytest.raises(NotImplementedError):
            _ = get_procedure("unknown", "cifar100")
        with pytest.raises(NotImplementedError):
            _ = get_procedure("resnet18", "unknown")
        with pytest.raises(NotImplementedError):
            _ = get_procedure("resnet50", "unknown")
        with pytest.raises(NotImplementedError):
            _ = get_procedure("wideresnet28x10", "unknown")
