# fmt: off
from pathlib import Path

import torch.nn as nn
from cli_test_helpers import ArgvContext

from torch_uncertainty import cls_main, init_args
from torch_uncertainty.baselines import ResNet, WideResNet
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.optimization_procedures import (
    optim_cifar10_resnet18,
    optim_cifar10_wideresnet,
)


# fmt: on
class TestCLI:
    """Testing the CLI function."""

    def test_cls_main_resnet(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext(""):
            args = init_args(ResNet, CIFAR10DataModule)

            # datamodule
            args.root = str(root / "data")
            dm = CIFAR10DataModule(**vars(args))

            # Simulate that summary is True & the only argument
            args.summary = True

            model = ResNet(
                num_classes=dm.num_classes,
                in_channels=dm.num_channels,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet18,
                style="cifar",
                **vars(args),
            )

            cls_main(model, dm, root, "std", args)

    def test_cls_main_other_arguments(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext("--seed 42 --max_epochs 1 --channels_last"):
            args = init_args(ResNet, CIFAR10DataModule)

            # datamodule
            args.root = root / "data"
            dm = CIFAR10DataModule(**vars(args))

            # Simulate that summary is True & the only argument
            args.summary = True

            model = ResNet(
                num_classes=dm.num_classes,
                in_channels=dm.num_channels,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet18,
                style="cifar",
                **vars(args),
            )

            cls_main(model, dm, root, "std", args)

    def test_cls_main_wideresnet(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext(""):
            args = init_args(WideResNet, CIFAR10DataModule)

            # datamodule
            args.root = root / "data"
            dm = CIFAR10DataModule(**vars(args))

            args.summary = True

            model = WideResNet(
                num_classes=dm.num_classes,
                in_channels=dm.num_channels,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_wideresnet,
                style="cifar",
                **vars(args),
            )

            cls_main(model, dm, root, "std", args)
