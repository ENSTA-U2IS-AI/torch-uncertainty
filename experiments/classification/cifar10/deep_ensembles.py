from pathlib import Path

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines import DeepEnsembles
from torch_uncertainty.datamodules import CIFAR10DataModule

if __name__ == "__main__":
    args = init_args(DeepEnsembles, CIFAR10DataModule)
    if args.root == "./data/":
        root = Path(__file__).parent.absolute().parents[2]
    else:
        root = Path(args.root)

    net_name = f"de-{args.backbone}-cifar10"

    # datamodule
    args.root = str(root / "data")
    dm = CIFAR10DataModule(**vars(args))

    # model
    args.task = "classification"
    model = DeepEnsembles(
        **vars(args),
        num_classes=dm.num_classes,
        in_channels=dm.num_channels,
    )

    args.test = -1

    cli_main(model, dm, root, net_name, args)
