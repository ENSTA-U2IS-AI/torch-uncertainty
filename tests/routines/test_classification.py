# fmt:off
from functools import partial
from pathlib import Path

import pytest
from cli_test_helpers import ArgvContext
from torch import nn

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.losses import DECLoss, ELBOLoss
from torch_uncertainty.optimization_procedures import optim_cifar10_resnet18
from torch_uncertainty.routines.classification import (
    ClassificationEnsemble,
    ClassificationSingle,
)

from .._dummies import (
    DummyClassificationBaseline,
    DummyClassificationDataModule,
)



class TestClassificationSingle:
    """Testing the classification routine with a single model."""

    def test_cli_main_dummy_binary(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext("file.py"):
            args = init_args(
                DummyClassificationBaseline, DummyClassificationDataModule
            )

            args.root = str(root / "data")
            dm = DummyClassificationDataModule(num_classes=1, **vars(args))

            model = DummyClassificationBaseline(
                num_classes=dm.num_classes,
                in_channels=dm.num_channels,
                loss=nn.BCEWithLogitsLoss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="single",
                **vars(args),
            )
            cli_main(model, dm, root, "dummy", args)

        with ArgvContext("file.py", "--logits"):
            args = init_args(
                DummyClassificationBaseline, DummyClassificationDataModule
            )

            args.root = str(root / "data")
            dm = DummyClassificationDataModule(num_classes=1, **vars(args))

            model = DummyClassificationBaseline(
                num_classes=dm.num_classes,
                in_channels=dm.num_channels,
                loss=nn.BCEWithLogitsLoss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="single",
                **vars(args),
            )
            cli_main(model, dm, root, "dummy", args)

    def test_cli_main_dummy_ood(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext("file.py", "--fast_dev_run"):
            args = init_args(
                DummyClassificationBaseline, DummyClassificationDataModule
            )

            args.root = str(root / "data")
            dm = DummyClassificationDataModule(**vars(args))
            loss = partial(
                ELBOLoss,
                criterion=nn.CrossEntropyLoss(),
                kl_weight=1e-5,
                num_samples=2,
            )
            model = DummyClassificationBaseline(
                num_classes=dm.num_classes,
                in_channels=dm.num_channels,
                loss=loss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="single",
                **vars(args),
            )
            cli_main(model, dm, root, "dummy", args)

        with ArgvContext(
            "file.py",
            "--evaluate_ood",
            "--entropy",
        ):
            args = init_args(
                DummyClassificationBaseline, DummyClassificationDataModule
            )

            args.root = str(root / "data")
            dm = DummyClassificationDataModule(**vars(args))

            model = DummyClassificationBaseline(
                num_classes=dm.num_classes,
                in_channels=dm.num_channels,
                loss=DECLoss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="single",
                **vars(args),
            )
            cli_main(model, dm, root, "dummy", args)

        with ArgvContext(
            "file.py", "--evaluate_ood", "--entropy", "--cutmix", "0.5"
        ):
            args = init_args(
                DummyClassificationBaseline, DummyClassificationDataModule
            )

            args.root = str(root / "data")
            dm = DummyClassificationDataModule(**vars(args))

            model = DummyClassificationBaseline(
                num_classes=dm.num_classes,
                in_channels=dm.num_channels,
                loss=DECLoss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="single",
                **vars(args),
            )
            with pytest.raises(NotImplementedError):
                cli_main(model, dm, root, "dummy", args)

    def test_classification_failures(self):
        with pytest.raises(ValueError):
            ClassificationSingle(
                10, nn.Module(), None, None, use_entropy=True, use_logits=True
            )

        with pytest.raises(ValueError):
            ClassificationSingle(10, nn.Module(), None, None, cutmix_alpha=-1)


class TestClassificationEnsemble:
    """Testing the classification routine with an ensemble model."""

    def test_cli_main_dummy_binary(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext("file.py"):
            args = init_args(
                DummyClassificationBaseline, DummyClassificationDataModule
            )

            # datamodule
            args.root = str(root / "data")
            dm = DummyClassificationDataModule(num_classes=1, **vars(args))
            loss = partial(
                ELBOLoss,
                criterion=nn.CrossEntropyLoss(),
                kl_weight=1e-5,
                num_samples=1,
            )
            model = DummyClassificationBaseline(
                num_classes=dm.num_classes,
                in_channels=dm.num_channels,
                loss=loss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="ensemble",
                **vars(args),
            )

            cli_main(model, dm, root, "dummy", args)

        with ArgvContext("file.py", "--mutual_information"):
            args = init_args(
                DummyClassificationBaseline, DummyClassificationDataModule
            )

            # datamodule
            args.root = str(root / "data")
            dm = DummyClassificationDataModule(num_classes=1, **vars(args))

            model = DummyClassificationBaseline(
                num_classes=dm.num_classes,
                in_channels=dm.num_channels,
                loss=nn.BCEWithLogitsLoss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="ensemble",
                **vars(args),
            )

            cli_main(model, dm, root, "dummy", args)

    def test_cli_main_dummy_ood(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext("file.py", "--logits"):
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

        with ArgvContext("file.py", "--evaluate_ood", "--entropy"):
            args = init_args(
                DummyClassificationBaseline, DummyClassificationDataModule
            )

            # datamodule
            args.root = str(root / "data")
            dm = DummyClassificationDataModule(**vars(args))

            model = DummyClassificationBaseline(
                num_classes=dm.num_classes,
                in_channels=dm.num_channels,
                loss=DECLoss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="ensemble",
                **vars(args),
            )

            cli_main(model, dm, root, "dummy", args)

        with ArgvContext("file.py", "--evaluate_ood", "--variation_ratio"):
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

    def test_classification_failures(self):
        with pytest.raises(ValueError):
            ClassificationEnsemble(
                10,
                nn.Module(),
                None,
                None,
                2,
                use_entropy=True,
                use_variation_ratio=True,
            )
