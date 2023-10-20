# fmt: off
from pathlib import Path

from torch import nn, optim

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines import ResNet
from torch_uncertainty.datamodules import TinyImageNetDataModule
from torch_uncertainty.utils import csv_writter


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
    args = init_args(ResNet, TinyImageNetDataModule)
    if args.root == "./data/":
        root = Path(__file__).parent.absolute().parents[2]
    else:
        root = Path(args.root)

    # net_name = f"{args.version}-resnet{args.arch}-tiny-imagenet"
    if args.exp_name == "":
        args.exp_name = f"{args.version}-resnet{args.arch}-cifar10"

    # datamodule
    args.root = str(root / "data")
    dm = TinyImageNetDataModule(**vars(args))

    if args.opt_temp_scaling:
        args.calibration_set = dm.get_test_set
    elif args.val_temp_scaling:
        args.calibration_set = dm.get_val_set
    else:
        args.calibration_set = None

    if args.use_cv:
        list_dm = dm.make_cross_val_splits(args.n_splits, args.train_over)
        list_model = []
        for i in range(len(list_dm)):
            list_model.append(
                ResNet(
                    num_classes=list_dm[i].dm.num_classes,
                    in_channels=list_dm[i].dm.num_channels,
                    loss=nn.CrossEntropyLoss,
                    optimization_procedure=optim_tiny,
                    style="cifar",
                    **vars(args),
                )
            )

        results = cli_main(
            list_model, list_dm, args.exp_dir, args.exp_name, args
        )
    else:
        # model
        model = ResNet(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=optim_tiny,
            style="cifar",
            **vars(args),
        )

        results = cli_main(model, dm, args.exp_dir, args.exp_name, args)

    for dict_result in results:
        csv_writter(
            Path(args.exp_dir) / Path(args.exp_name) / "results.csv",
            dict_result,
        )
