# fmt: off
from pathlib import Path

import torch.nn as nn
from cli_test_helpers import ArgvContext

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines import VGG, ResNet, WideResNet
from torch_uncertainty.baselines.regression import MLP
from torch_uncertainty.datamodules import CIFAR10DataModule, UCIDataModule
from torch_uncertainty.optimization_procedures import (
    optim_cifar10_resnet18,
    optim_cifar10_vgg16,
    optim_cifar10_wideresnet,
    optim_regression,
)


# fmt: on
class TestCLI:
    """Testing the CLI function."""

    def test_cli_main_resnet(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext("--style cifar"):
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
                **vars(args),
            )

            cli_main(model, dm, root, "std", args)

    def test_cli_main_other_arguments(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext(
            "--seed 42 --max_epochs 1 --channels_last --style cifar"
        ):
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
                **vars(args),
            )

            cli_main(model, dm, root, "std", args)

    def test_cli_main_wideresnet(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext("--style cifar"):
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
                **vars(args),
            )

            cli_main(model, dm, root, "std", args)

    def test_cli_main_vgg(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext("--style cifar"):
            args = init_args(VGG, CIFAR10DataModule)

            # datamodule
            args.root = root / "data"
            dm = CIFAR10DataModule(**vars(args))

            args.summary = True

            model = VGG(
                num_classes=dm.num_classes,
                in_channels=dm.num_channels,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_vgg16,
                **vars(args),
            )

            cli_main(model, dm, root, "std", args)

    def test_cli_main_mlp(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext(""):
            args = init_args(MLP, UCIDataModule)

            # datamodule
            args.root = root / "data"
            dm = UCIDataModule(
                dataset_name="kin8nm", input_shape=(1, 5), **vars(args)
            )

            args.summary = True

            model = MLP(
                num_outputs=1,
                in_features=5,
                hidden_dims=[],
                dist_estimation=False,
                loss=nn.MSELoss,
                optimization_procedure=optim_regression,
                **vars(args),
            )

            cli_main(model, dm, root, "std", args)
