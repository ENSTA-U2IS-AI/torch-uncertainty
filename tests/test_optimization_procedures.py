# flake8: noqa
# fmt: off
from torch_uncertainty.models.resnet import resnet18, resnet50

# from torch_uncertainty.models.wideresnet import resnet18, resnet50
from torch_uncertainty.optimization_procedures import *


# fmt: on
class TestOptProcedures:
    def test_optim_cifar10_resnet18(self):
        model = resnet18(in_channels=3, num_classes=10)
        optim_cifar10_resnet18(model)

    def test_optim_cifar10_resnet50(self):
        model = resnet50(in_channels=3, num_classes=10)
        optim_cifar10_resnet50(model)

    # def test_optim_cifar10_wideresnet(self):
    #     model = resnet50()
    #     optim_cifar10_wideresnet(model)

    def test_optim_cifar100_resnet18(self):
        model = resnet18(in_channels=3, num_classes=100)
        optim_cifar100_resnet18(model)

    def test_optim_cifar100_resnet50(self):
        model = resnet50(in_channels=3, num_classes=100)
        optim_cifar100_resnet50(model)

    # def test_optim_cifar100_wideresnet(self):
    #     model = resnet50()
    #     optim_cifar10_wideresnet(model)

    def test_optim_imagenet_resnet50(self):
        model = resnet50(in_channels=3, num_classes=1000)
        optim_imagenet_resnet50(model)

    def test_optim_imagenet_resnet50_A3(self):
        model = resnet50(in_channels=3, num_classes=1000)
        optim_imagenet_resnet50_A3(model)
