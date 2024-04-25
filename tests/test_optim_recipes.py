# ruff: noqa: F401
import pytest

from torch_uncertainty.models.resnet import resnet
from torch_uncertainty.models.vgg import vgg16
from torch_uncertainty.models.wideresnet import wideresnet28x10
from torch_uncertainty.optim_recipes import (
    get_procedure,
)


class TestOptProcedures:
    def test_optim_cifar10(self):
        model = resnet(in_channels=3, num_classes=10, arch=18)
        get_procedure("resnet18", "cifar10", "standard")(model)
        get_procedure("resnet34", "cifar10", "masked")(model)
        get_procedure("resnet50", "cifar10", "packed")(model)
        get_procedure("wideresnet28x10", "cifar10", "batched")(model)
        get_procedure("vgg16", "cifar10", "standard")(model)

    def test_optim_cifar100(self):
        model = resnet(in_channels=3, num_classes=10, arch=18)
        get_procedure("resnet18", "cifar100", "masked")(model)
        get_procedure("resnet34", "cifar100", "masked")(model)
        get_procedure("resnet50", "cifar100")(model)
        get_procedure("wideresnet28x10", "cifar100")(model)
        get_procedure("vgg16", "cifar100", "standard")(model)

    def test_optim_tinyimagenet(self):
        model = resnet(in_channels=3, num_classes=10, arch=18)
        get_procedure("resnet34", "tiny-imagenet", "standard")(model)
        get_procedure("resnet50", "tiny-imagenet", "standard")(model)

    def test_optim_imagenet_resnet50(self):
        model = resnet(in_channels=3, num_classes=10, arch=18)
        get_procedure("resnet50", "imagenet", "standard", "A3")(model)
        get_procedure("resnet50", "imagenet", "standard")(model)

    def test_optim_unknown(self):
        with pytest.raises(NotImplementedError):
            _ = get_procedure("unknown", "cifar100")
        with pytest.raises(NotImplementedError):
            _ = get_procedure("resnet18", "unknown")
        with pytest.raises(NotImplementedError):
            _ = get_procedure("resnet34", "unknown")
        with pytest.raises(NotImplementedError):
            _ = get_procedure("resnet50", "unknown")
        with pytest.raises(NotImplementedError):
            _ = get_procedure("wideresnet28x10", "unknown")
