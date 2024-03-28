from pathlib import Path

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines import DeepEnsemblesBaseline
from torch_uncertainty.datamodules import CIFAR100DataModule

if __name__ == "__main__":
    args = init_args(DeepEnsemblesBaseline, CIFAR100DataModule)
    if args.root == "./data/":
        root = Path(__file__).parent.absolute().parents[2]
    else:
        root = Path(args.root)

    net_name = f"de-{args.backbone}-cifar100"

    # datamodule
    args.root = str(root / "data")
    dm = CIFAR100DataModule(**vars(args))

    # model
    args.task = "classification"
    model = DeepEnsemblesBaseline(
        **vars(args),
        num_classes=dm.num_classes,
        in_channels=dm.num_channels,
    )

    args.test = -1

    cli_main(model, dm, root, net_name, args)
