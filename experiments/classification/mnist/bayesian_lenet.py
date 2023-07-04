# fmt: off
from functools import partial
from pathlib import Path

import torch.nn as nn
import torch.optim as optim

from torch_uncertainty import cls_main, init_args
from torch_uncertainty.datamodules import MNISTDataModule
from torch_uncertainty.losses import ELBOLoss
from torch_uncertainty.models.lenet import bayesian_lenet
from torch_uncertainty.routines.classification import ClassificationSingle

# fmt: on


def optim_lenet(model: nn.Module) -> dict:
    """Optimization procedure for LeNet.

    Uses Adam default hyperparameters.

    Args:
        model (nn.Module): LeNet model.
    """
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer=optimizer, T_max=100, eta_min=1e-6
    # )
    return {"optimizer": optimizer}  # , "scheduler": scheduler}


if __name__ == "__main__":
    root = Path(__file__).parent.absolute().parents[2]

    args = init_args(datamodule=MNISTDataModule)

    net_name = "bayesian-lenet-mnist"

    # datamodule
    args.root = str(root / "data")
    dm = MNISTDataModule(**vars(args))

    # model
    model = bayesian_lenet(dm.num_channels, dm.num_classes)

    # Here, the loss is a bit more complicated
    #   hyperparameters are from blitz.
    loss = partial(
        ELBOLoss,
        model=model,
        criterion=nn.CrossEntropyLoss(),
        kl_weight=1 / 50000,
        num_samples=3,
    )

    baseline = ClassificationSingle(
        model=model,
        num_classes=dm.num_classes,
        in_channels=dm.num_channels,
        loss=loss,
        optimization_procedure=optim_lenet,
        **vars(args),
    )

    cls_main(baseline, dm, root, net_name, args)
