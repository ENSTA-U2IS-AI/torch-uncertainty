from functools import partial
from pathlib import Path

from torch import nn, optim

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.datamodules import MNISTDataModule
from torch_uncertainty.losses import ELBOLoss
from torch_uncertainty.models.lenet import bayesian_lenet
from torch_uncertainty.routines.classification import ClassificationSingle


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
    return {"optimizer": optimizer}


if __name__ == "__main__":
    args = init_args(datamodule=MNISTDataModule)
    if args.root == "./data/":
        root = Path(__file__).parent.absolute().parents[2]
    else:
        root = Path(args.root)

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

    cli_main(baseline, dm, "logs/", net_name, args)
