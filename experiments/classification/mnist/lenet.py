from pathlib import Path

from torch import nn, optim

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.datamodules import MNISTDataModule
from torch_uncertainty.models.lenet import lenet
from torch_uncertainty.routines.classification import ClassificationSingle


def optim_lenet(model: nn.Module) -> dict:
    """Optimization procedure for LeNet.

    Uses Adam default hyperparameters.

    Args:
        model (nn.Module): LeNet model.
    """
    return {
        "optimizer": optim.Adam(
            model.parameters(),
        )
    }


if __name__ == "__main__":
    args = init_args(datamodule=MNISTDataModule)
    if args.root == "./data/":
        root = Path(__file__).parent.absolute().parents[2]
    else:
        root = Path(args.root)

    if args.exp_name == "":
        args.exp_name = "std-lenet-mnist"

    # datamodule
    args.root = str(root / "data")
    dm = MNISTDataModule(**vars(args))

    # model
    model = lenet(dm.num_channels, dm.num_classes)

    baseline = ClassificationSingle(
        model=model,
        num_classes=dm.num_classes,
        in_channels=dm.num_channels,
        loss=nn.CrossEntropyLoss,
        optimization_procedure=optim_lenet,
        **vars(args),
    )

    cli_main(baseline, dm, args.exp_dir, args.exp_name, args)
