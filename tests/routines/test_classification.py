# fmt:off
from pathlib import Path

import torch.nn as nn
from cli_test_helpers import ArgvContext

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.optimization_procedures import optim_cifar10_resnet18

from .._dummies import (
    DummyClassificationBaseline,
    DummyClassificationDataModule,
)


# fmt:on
class TestClassificationSingle:
    """Testing the classification routine with a single model."""

    def test_cli_main_dummy_binary(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext(""):
            args = init_args(
                DummyClassificationBaseline, DummyClassificationDataModule
            )

            # datamodule
            args.root = str(root / "data")
            dm = DummyClassificationDataModule(num_classes=1, **vars(args))

            model = DummyClassificationBaseline(
                num_classes=dm.num_classes,
                in_channels=dm.num_channels,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="single",
                **vars(args),
            )

            cli_main(model, dm, root, "dummy", args)

    def test_cli_main_dummy_ood(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext("--evaluate_ood"):
            args = init_args(
                DummyClassificationBaseline, DummyClassificationDataModule
            )

            # datamodule
            args.root = str(root / "data")
            dm = DummyClassificationDataModule(**vars(args))

            model = DummyClassificationBaseline(
                num_classes=dm.num_classes,
                in_channels=dm.num_channels,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="single",
                **vars(args),
            )

            cli_main(model, dm, root, "dummy", args)


class TestClassificationEnsemble:
    """Testing the classification routine with an ensemble model."""

    def test_cli_main_dummy_binary(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext(""):
            args = init_args(
                DummyClassificationBaseline, DummyClassificationDataModule
            )

            # datamodule
            args.root = str(root / "data")
            dm = DummyClassificationDataModule(num_classes=1, **vars(args))

            model = DummyClassificationBaseline(
                num_classes=dm.num_classes,
                in_channels=dm.num_channels,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="ensemble",
                **vars(args),
            )

            cli_main(model, dm, root, "dummy", args)

    def test_cli_main_dummy_ood(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext("--evaluate_ood"):
            args = init_args(
                DummyClassificationBaseline, DummyClassificationDataModule
            )

            # datamodule
            args.root = str(root / "data")
            dm = DummyClassificationDataModule(**vars(args))

            model = DummyClassificationBaseline(
                num_classes=dm.num_classes,
                in_channels=dm.num_channels,
                loss=nn.CrossEntropyLoss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="ensemble",
                **vars(args),
            )

            cli_main(model, dm, root, "dummy", args)
