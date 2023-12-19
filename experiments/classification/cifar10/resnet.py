from pathlib import Path

from torch import nn

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines import ResNet
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.optimization_procedures import get_procedure
from torch_uncertainty.utils import csv_writer

if __name__ == "__main__":
    args = init_args(ResNet, CIFAR10DataModule)
    if args.root == "./data/":
        root = Path(__file__).parent.absolute().parents[2]
    else:
        root = Path(args.root)

    if args.exp_name == "":
        args.exp_name = f"{args.version}-resnet{args.arch}-cifar10"

    # datamodule
    args.root = str(root / "data")
    dm = CIFAR10DataModule(**vars(args))

    if args.opt_temp_scaling:
        calibration_set = dm.get_test_set
    elif args.val_temp_scaling:
        calibration_set = dm.get_val_set
    else:
        calibration_set = None

    results = None
    if args.use_cv:
        list_dm = dm.make_cross_val_splits(args.n_splits, args.train_over)
        list_model = [
            ResNet(
                num_classes=list_dm[i].dm.num_classes,
                in_channels=list_dm[i].dm.num_channels,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=get_procedure(
                    f"resnet{args.arch}", "cifar10", args.version
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
        model = ResNet(
            num_classes=dm.num_classes,
            in_channels=dm.num_channels,
            loss=nn.CrossEntropyLoss,
            optimization_procedure=get_procedure(
                f"resnet{args.arch}", "cifar10", args.version
            ),
            style="cifar",
            calibration_set=calibration_set,
            **vars(args),
        )

        results = cli_main(model, dm, args.exp_dir, args.exp_name, args)

    if results is not None:
        for dict_result in results:
            csv_writer(
                Path(args.exp_dir) / Path(args.exp_name) / "results.csv",
                dict_result,
            )
