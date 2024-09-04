import logging
from collections.abc import Callable
from functools import partial
from typing import Literal

import torch
from timm.optim import Lamb
from torch import nn, optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    LRScheduler,
    MultiStepLR,
    SequentialLR,
)


def optim_abnn(
    model: nn.Module,
    lr: float,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    nesterov: bool = True,
) -> dict:
    """ABNN finetuning recipe."""
    optimizer = optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=nesterov,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[1, 4],
        gamma=0.1,
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def optim_cifar10_resnet18(
    model: nn.Module,
) -> dict[str, Optimizer | LRScheduler]:
    """Optimizer to train a ResNet18 on CIFAR-10."""
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[25, 50],
        gamma=0.1,
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def optim_cifar10_resnet50(
    model: nn.Module,
) -> dict[str, Optimizer | LRScheduler]:
    r"""Hyperparameters from Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf.
    """
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[60, 120, 160],
        gamma=0.2,
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def optim_cifar10_wideresnet(
    model: nn.Module,
) -> dict[str, Optimizer | LRScheduler]:
    """Optimizer to train a WideResNet28x10 on CIFAR-10."""
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[60, 120, 160],
        gamma=0.2,
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def optim_cifar10_vgg16(
    model: nn.Module,
) -> dict[str, Optimizer | LRScheduler]:
    """Optimizer to train a VGG16 on CIFAR-10."""
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.005,
        weight_decay=1e-6,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[25, 50],
        gamma=0.1,
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def optim_cifar100_resnet18(
    model: nn.Module,
) -> dict[str, Optimizer | LRScheduler]:
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[25, 50],
        gamma=0.1,
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def optim_cifar100_resnet50(
    model: nn.Module,
) -> dict[str, Optimizer | LRScheduler]:
    r"""Hyperparameters from Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf.
    """
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[60, 120, 160],
        gamma=0.2,
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def optim_cifar100_vgg16(
    model: nn.Module,
) -> dict[str, Optimizer | LRScheduler]:
    """Optimizer to train a VGG16 on CIFAR-100."""
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.05,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[60, 120, 160],
        gamma=0.2,
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def optim_imagenet_resnet50(
    model: nn.Module,
    num_epochs: int = 90,
    start_lr: float = 0.256,
    end_lr: float = 0,
) -> dict:
    r"""Hyperparameters from Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf.
    """
    optimizer = optim.SGD(
        model.parameters(),
        lr=start_lr,
        momentum=0.875,
        weight_decay=3.0517578125e-05,
        nesterov=False,
    )
    scheduler = CosineAnnealingLR(optimizer, num_epochs, eta_min=end_lr)
    return {
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "monitor": "cls/val_acc",
    }


def optim_imagenet_resnet50_a3(
    model: nn.Module, effective_batch_size: int | None = None
) -> dict:
    """Training procedure proposed in ResNet strikes back: An improved training
        procedure in timm.

    Args:
        model (nn.Module): The model to be optimized.
        effective_batch_size (int, optional): The batch size of the model
            (taking multiple GPUs into account). Defaults to None.

    Returns:
        dict: The optimizer and the scheduler for the training.
    """
    if effective_batch_size is None:
        logging.warning(
            "Setting effective batch size to 2048 for steps computations !"
        )
        effective_batch_size = 2048

    optimizer = Lamb(model.parameters(), lr=0.008, weight_decay=0.02)

    warmup = LinearLR(
        optimizer,
        start_factor=1e-4,
        end_factor=1,
        total_iters=5 * (1281167 // effective_batch_size + 1),
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        eta_min=1e-6,
        T_max=105 * (1281167 // effective_batch_size + 1),
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine_scheduler],
        milestones=[5 * (1281167 // effective_batch_size + 1)],
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
        },
        "monitor": "cls/val_acc",
    }


def optim_cifar10_resnet34(
    model: nn.Module,
) -> dict[str, Optimizer | LRScheduler]:
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[100, 150],
        gamma=0.1,
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def optim_cifar100_resnet34(
    model: nn.Module,
) -> dict[str, Optimizer | LRScheduler]:
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[100, 150],
        gamma=0.1,
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def optim_tinyimagenet_resnet34(
    model: nn.Module,
) -> dict[str, Optimizer | LRScheduler]:
    """Optimization recipe from 'The Devil is in the Margin: Margin-based
    Label Smoothing for Network Calibration',
    (CVPR 2022, https://arxiv.org/abs/2111.15430):
    'We train for 100 epochs with a learning rate of 0.1 for the first
    40 epochs, of 0.01 for the next 20 epochs and of 0.001 for the last
    40 epochs.'.
    """
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[40, 60],
        gamma=0.1,
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def optim_tinyimagenet_resnet50(
    model: nn.Module,
) -> dict[str, Optimizer | LRScheduler]:
    """Optimization recipe from 'The Devil is in the Margin: Margin-based
    Label Smoothing for Network Calibration',
    (CVPR 2022, https://arxiv.org/abs/2111.15430):
    'We train for 100 epochs with a learning rate of 0.1 for the first
    40 epochs, of 0.01 for the next 20 epochs and of 0.001 for the last
    40 epochs.'.
    """
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )
    scheduler = MultiStepLR(
        optimizer,
        milestones=[40, 60],
        gamma=0.1,
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def batch_ensemble_wrapper(model: nn.Module, optim_recipe: Callable) -> dict:
    procedure = optim_recipe(model)
    param_optimizer = procedure["optimizer"]
    scheduler = procedure["lr_scheduler"]

    weight_decay = param_optimizer.defaults["weight_decay"]
    lr = param_optimizer.defaults["lr"]
    momentum = param_optimizer.defaults["momentum"]

    name_list = ["R", "S"]
    params_multi_tmp = list(
        filter(
            lambda kv: (name_list[0] in kv[0]) or (name_list[1] in kv[0]),
            model.named_parameters(),
        )
    )
    param_core_tmp = list(
        filter(
            lambda kv: (name_list[0] not in kv[0])
            and (name_list[1] not in kv[0]),
            model.named_parameters(),
        )
    )

    params_multi = [param for _, param in params_multi_tmp]
    param_core = [param for _, param in param_core_tmp]
    optimizer = optim.SGD(
        [
            {"params": param_core, "weight_decay": weight_decay},
            {"params": params_multi, "weight_decay": 0.0},
        ],
        lr=lr,
        momentum=momentum,
    )

    scheduler.optimizer = optimizer
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def get_procedure(
    arch_name: str,
    ds_name: str,
    method: str = "",
    imagenet_recipe: str | None = None,
) -> Callable:
    """Get the optimization recipe for a given architecture and dataset.

    Args:
        arch_name (str): The name of the architecture.
        ds_name (str): The name of the dataset.
        method (str, optional): The name of the method. Defaults to "".
        imagenet_recipe (str, optional): The recipe to use for
            ImageNet. Defaults to None.

    Returns:
        callable: The optimization recipe.
    """
    if arch_name in ["resnet18", "resnet20"]:
        if ds_name == "cifar10":
            procedure = optim_cifar10_resnet18
        elif ds_name == "cifar100":
            procedure = optim_cifar100_resnet18
        else:
            raise NotImplementedError(f"Dataset {ds_name} not implemented.")
    elif arch_name == "resnet34":
        if ds_name == "cifar10":
            procedure = optim_cifar10_resnet34
        elif ds_name == "cifar100":
            procedure = optim_cifar100_resnet34
        elif ds_name == "tiny-imagenet":
            procedure = optim_tinyimagenet_resnet34
        else:
            raise NotImplementedError(f"No recipe for dataset: {ds_name}.")
    elif arch_name == "resnet50":
        if ds_name == "cifar10":
            procedure = optim_cifar10_resnet50
        elif ds_name == "cifar100":
            procedure = optim_cifar100_resnet50
        elif ds_name == "tiny-imagenet":
            procedure = optim_tinyimagenet_resnet50
        elif ds_name == "imagenet":
            if imagenet_recipe is not None and imagenet_recipe == "A3":
                procedure = optim_imagenet_resnet50_a3
            else:
                procedure = optim_imagenet_resnet50
        else:
            raise NotImplementedError(f"No recipe for dataset: {ds_name}.")
    elif arch_name == "wideresnet28x10":
        if ds_name in ("cifar10", "cifar100"):
            procedure = optim_cifar10_wideresnet
        else:
            raise NotImplementedError(f"No recipe for dataset: {ds_name}.")
    elif "vgg" in arch_name:
        if ds_name == "cifar10":
            procedure = optim_cifar10_vgg16
        elif ds_name == "cifar100":
            procedure = optim_cifar100_vgg16
        else:
            raise NotImplementedError(f"No recipe for dataset: {ds_name}.")
    else:
        raise NotImplementedError(f"No recipe for architecture: {arch_name}.")

    if method == "batched":
        procedure = partial(batch_ensemble_wrapper, optim_recipe=procedure)

    return procedure


class WarmupScheduler(SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        base_scheduler: type[LRScheduler],
        warmup_start_factor: float,
        warmup_epochs: int,
        scheduler_args: dict[str, float],
    ) -> None:
        """Scheduler with linear warmup.

        Args:
            optimizer (Optimizer): The optimizer to be used.*
            base_scheduler (type[LRScheduler]): The base scheduler class to use after
                the warmup.
            warmup_start_factor (float): The multiplicative factor to apply to
                the learning rate at the start of the warmup.
            warmup_epochs (int): The number of epochs to warmup the learning
                rate.
            scheduler_args (dict[str, float]): The arguments to pass to the base
                scheduler.
        """
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1,
            total_iters=warmup_epochs,
        )
        base_scheduler = base_scheduler(optimizer, **scheduler_args)
        super().__init__(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, base_scheduler],
            milestones=[warmup_epochs],
        )


class CosineAnnealingWarmup(WarmupScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        warmup_start_factor: float,
        warmup_epochs: int,
        max_epochs: int,
        eta_min: float = 0,
    ) -> None:
        """Cosine annealing scheduler with linear warmup.

        Args:
            optimizer (Optimizer): The optimizer to be used.
            warmup_start_factor (float): The multiplicative factor to apply to
                the learning rate at the start of the warmup.
            warmup_epochs (int): The number of epochs to warmup the learning
                rate.
            max_epochs (int): The total number of epochs including warmup.
            eta_min (float): The minimum learning rate.
        """
        super().__init__(
            optimizer=optimizer,
            base_scheduler=CosineAnnealingLR,
            warmup_start_factor=warmup_start_factor,
            warmup_epochs=warmup_epochs,
            scheduler_args={
                "T_max": max_epochs - warmup_epochs,
                "eta_min": eta_min,
            },
        )


class CosineSWALR(SequentialLR):
    def __init__(
        self,
        optimizer: Optimizer,
        milestone: int,
        swa_lr: float,
        anneal_epochs: int,
        optim_eta_min: float = 0,
        anneal_strategy: Literal["cos", "linear"] = "cos",
    ) -> None:
        """Chains a Cosine scheduler and a SWA scheduler.

        This class is an example of a wrapper to enable training SWA and SWAG
        models using the CLI. You may create your own class following this
        example.

        Args:
            optimizer (Optimizer): The optimizer to be used.
            milestone (int): The epoch to start the SWA.
            swa_lr (float): The learning rate to use for the SWA model.
            anneal_epochs (int): The number of epochs to anneal the learning rate.
            optim_eta_min (float): The minimum learning rate for the first optimizer.
            anneal_strategy (Literal["cos", "linear"]): The strategy to anneal the learning rate.
        """
        optim_scheduler = CosineAnnealingLR(
            optimizer=optimizer, T_max=milestone, eta_min=optim_eta_min
        )
        swa_scheduler = torch.optim.swa_utils.SWALR(
            optimizer,
            swa_lr=swa_lr,
            anneal_epochs=anneal_epochs,
            anneal_strategy=anneal_strategy,
        )
        super().__init__(
            optimizer=optimizer,
            schedulers=[optim_scheduler, swa_scheduler],
            milestones=[milestone],
        )
