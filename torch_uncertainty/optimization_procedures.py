# fmt: off
import torch.nn as nn
import torch.optim as optim
from timm.optim import Lamb


# fmt:on
def optim_cifar10_resnet18(model: nn.Module) -> dict:
    """optimizer to train a ResNet18 on CIFAR-10"""
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.05,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[25, 50],
        gamma=0.1,
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def optim_cifar10_resnet50(model: nn.Module) -> dict:
    r"""Hyperparameters from Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    """
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 120, 160],
        gamma=0.2,
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def optim_cifar10_wideresnet(model: nn.Module) -> dict:
    """optimizer to train a WideResNet28x10 on CIFAR-10"""
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 120, 160],
        gamma=0.2,
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def optim_cifar100_resnet18(model: nn.Module) -> dict:
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 120, 160],
        gamma=0.2,
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def optim_cifar100_resnet50(model: nn.Module, adam: bool = False) -> dict:
    r"""Hyperparameters from Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    """
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[60, 120, 160],
        gamma=0.2,
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


def optim_imagenet_resnet50(
    model: nn.Module,
    n_epochs: int = 90,
    start_lr: float = 0.256,
    end_lr: float = 0,
) -> dict:
    r"""Hyperparameters from Deep Residual Learning for Image Recognition
    https://arxiv.org/pdf/1512.03385.pdf
    """
    optimizer = optim.SGD(
        model.parameters(),
        lr=start_lr,
        momentum=0.875,
        weight_decay=3.0517578125e-05,
        nesterov=False,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, n_epochs, eta_min=end_lr
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": scheduler,
        "monitor": "hp/val_acc",
    }


def optim_imagenet_resnet50_A3(
    model: nn.Module, effective_batch_size: int = None
) -> dict:
    """
    Training procedure proposed in ResNet strikes back: An improved training
        procedure in timm.

    Args:
        model (nn.Module): The model to be optimized.
        effective_batch_size (int, optional): The batch size of the model
            (taking multiple GPUs into account). Defaults to None.

    Returns:
        dict: The optimizer and the scheduler for the training.
    """
    if effective_batch_size is None:
        print("Setting effective batch size to 2048 for steps computations !")
        effective_batch_size = 2048

    optimizer = Lamb(model.parameters(), lr=0.008, weight_decay=0.02)

    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1e-4,
        end_factor=1,
        total_iters=5 * (1281167 // effective_batch_size + 1),
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        eta_min=1e-6,
        T_max=105 * (1281167 // effective_batch_size + 1),
    )
    scheduler = optim.lr_scheduler.SequentialLR(
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
        "monitor": "hp/val_acc",
    }
