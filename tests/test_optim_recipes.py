# ruff: noqa: F401
import pytest
import torch

from torch_uncertainty.optim_recipes import (
    CosineAnnealingWarmup,
    CosineSWALR,
    get_procedure,
    optim_abnn,
)


class TestCosineAnnealingWarmup:
    def test_full_cosine_annealing_warmup(self):
        CosineAnnealingWarmup(
            torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=1e-3),
            warmup_start_factor=0.1,
            warmup_epochs=5,
            max_epochs=100,
            eta_min=1e-5,
        )


class TestCosineSWALR:
    def test_full_swa_lr(self):
        CosineSWALR(
            torch.optim.SGD(torch.nn.Linear(1, 1).parameters(), lr=1e-3),
            swa_lr=1,
            milestone=12,
            anneal_epochs=5,
        )


class TestOptProcedures:
    def test_optim_cifar10(self):
        model = torch.nn.Linear(1, 1)
        get_procedure("resnet18", "cifar10", "standard")(model)
        get_procedure("resnet34", "cifar10", "masked")(model)
        get_procedure("resnet50", "cifar10", "packed")(model)
        get_procedure("wideresnet28x10", "cifar10", "batched")(model)
        get_procedure("vgg16", "cifar10", "standard")(model)
        optim_abnn(model, lr=0.1)

    def test_optim_cifar100(self):
        model = torch.nn.Linear(1, 1)
        get_procedure("resnet18", "cifar100", "masked")(model)
        get_procedure("resnet34", "cifar100", "masked")(model)
        get_procedure("resnet50", "cifar100")(model)
        get_procedure("wideresnet28x10", "cifar100")(model)
        get_procedure("vgg16", "cifar100", "standard")(model)

    def test_optim_tinyimagenet(self):
        model = torch.nn.Linear(1, 1)
        get_procedure("resnet34", "tiny-imagenet", "standard")(model)
        get_procedure("resnet50", "tiny-imagenet", "standard")(model)

    def test_optim_imagenet_resnet50(self):
        model = torch.nn.Linear(1, 1)
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
