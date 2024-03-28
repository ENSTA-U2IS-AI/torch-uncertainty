from pathlib import Path

from torch import nn, optim

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines import ResNetBaseline
from torch_uncertainty.datamodules import TinyImageNetDataModule
from torch_uncertainty.optim_recipes import get_procedure
from torch_uncertainty.utils import csv_writer


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
    args = init_args(ResNetBaseline, TinyImageNetDataModule)
    if args.root == "./data/":
        root = Path(__file__).parent.absolute().parents[2]
    else:
        root = Path(args.root)

    if args.exp_name == "":
        args.exp_name = f"{args.version}-resnet{args.arch}-tinyimagenet"

    # datamodule
    args.root = str(root / "data")
    dm = TinyImageNetDataModule(**vars(args))

    if args.opt_temp_scaling:
        calibration_set = dm.get_test_set
    elif args.val_temp_scaling:
        calibration_set = dm.get_val_set
    else:
        calibration_set = None

    if args.use_cv:
        list_dm = dm.make_cross_val_splits(args.n_splits, args.train_over)
        list_model = [
            ResNetBaseline(
                num_classes=list_dm[i].dm.num_classes,
                in_channels=list_dm[i].dm.num_channels,
                loss=nn.CrossEntropyLoss(),
                optim_recipe=get_procedure(
                    f"resnet{args.arch}", "tiny-imagenet", args.version
                ),
                style="cifar",
                calibration_set=calibration_set,
                **vars(args),
            )
            for i in range(len(list_dm))
        ]

        results = cli_main(
            list_model, list_dm, args.exp_dir, args.exp_name, args
        )
    else:
        # model
        model = ResNetBaseline(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss(),
            optim_recipe=get_procedure(
                f"resnet{args.arch}", "tiny-imagenet", args.version
            ),
            calibration_set=calibration_set,
            style="cifar",
            **vars(args),
        )

        results = cli_main(model, dm, args.exp_dir, args.exp_name, args)

    if results is not None:
        for dict_result in results:
            csv_writer(
                Path(args.exp_dir) / Path(args.exp_name) / "results.csv",
                dict_result,
            )
