# fmt: off
from pathlib import Path

import torch.nn as nn
import torch.optim as optim

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines import ResNet
from torch_uncertainty.datamodules import TinyImageNetDataModule


# fmt: on
def optim_tiny(model: nn.Module) -> dict:
    optimizer = optim.SGD(
        model.parameters(), lr=0.2, weight_decay=1e-4, momentum=0.9
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        eta_min=0,
        T_max=200,
    )
    return {"optimizer": optimizer, "lr_scheduler": scheduler}


if __name__ == "__main__":
    root = Path(__file__).parent.absolute().parents[2]

    args = init_args(ResNet, TinyImageNetDataModule)

    net_name = f"{args.version}-resnet{args.arch}-tiny-imagenet"

    # datamodule
    args.root = str(root / "data")
    dm = TinyImageNetDataModule(**vars(args))

    # model
    model = ResNet(
        num_classes=dm.num_classes,
        in_channels=dm.num_channels,
        loss=nn.CrossEntropyLoss,
        optimization_procedure=optim_tiny,
        style="cifar",
        **vars(args),
    )

    # torch.compile(model)

    cli_main(model, dm, root, net_name, args)
