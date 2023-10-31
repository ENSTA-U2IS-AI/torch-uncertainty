from pathlib import Path

from torch import nn

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines import WideResNet
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.optimization_procedures import get_procedure

if __name__ == "__main__":
    root = Path(__file__).parent.absolute().parents[2]

    args = init_args(WideResNet, CIFAR10DataModule)

    net_name = f"{args.version}-wideresnet28x10-cifar10"

    # datamodule
    args.root = str(root / "data")
    dm = CIFAR10DataModule(**vars(args))

    # model
    model = WideResNet(
        num_classes=dm.num_classes,
        in_channels=dm.num_channels,
        loss=nn.CrossEntropyLoss,
        optimization_procedure=get_procedure(
            "wideresnet28x10", "cifar10", args.version
        ),
        style="cifar",
        **vars(args),
    )

    cli_main(model, dm, root, net_name, args)
