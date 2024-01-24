from functools import partial
from pathlib import Path

import pytest
from cli_test_helpers import ArgvContext
from torch import nn

from tests._dummies import (
    DummyClassificationBaseline,
    DummyClassificationDataModule,
    DummyClassificationDataset,
    dummy_model,
)
from torch_uncertainty import cli_main, init_args
from torch_uncertainty.losses import DECLoss, ELBOLoss
from torch_uncertainty.optimization_procedures import optim_cifar10_resnet18
from torch_uncertainty.routines.classification import (
    ClassificationEnsemble,
    ClassificationSingle,
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
            args.eval_grouping_loss = True
            dm = DummyClassificationDataModule(
                num_classes=1, num_images=100, **vars(args)
            )

            model = DummyClassificationBaseline(
                num_classes=dm.num_classes,
                in_channels=dm.num_channels,
                loss=nn.BCEWithLogitsLoss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="single",
                **vars(args),
            )
            cli_main(model, dm, root, "logs/dummy", args)

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
            cli_main(model, dm, root, "logs/dummy", args)

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
            cli_main(model, dm, root, "logs/dummy", args)

        with ArgvContext(
            "file.py",
            "--eval-ood",
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
            cli_main(model, dm, root, "logs/dummy", args)

        with ArgvContext(
            "file.py",
            "--eval-ood",
            "--entropy",
            "--cutmix_alpha",
            "0.5",
            "--mixtype",
            "timm",
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
                cli_main(model, dm, root, "logs/dummy", args)

    def test_cli_main_dummy_mixup_ts_cv(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext(
            "file.py",
            "--mixtype",
            "kernel_warping",
            "--mixup_alpha",
            "1.",
            "--dist_sim",
            "inp",
            "--val_temp_scaling",
            "--use_cv",
        ):
            args = init_args(
                DummyClassificationBaseline, DummyClassificationDataModule
            )

            args.root = str(root / "data")
            dm = DummyClassificationDataModule(num_classes=10, **vars(args))
            dm.dataset = (
                lambda root,
                num_channels,
                num_classes,
                image_size,
                transform,
                num_images: DummyClassificationDataset(
                    root,
                    num_channels=num_channels,
                    num_classes=num_classes,
                    image_size=image_size,
                    transform=transform,
                    num_images=20,
                )
            )

            list_dm = dm.make_cross_val_splits(2, 1)
            list_model = [
                DummyClassificationBaseline(
                    num_classes=list_dm[i].dm.num_classes,
                    in_channels=list_dm[i].dm.num_channels,
                    loss=nn.CrossEntropyLoss,
                    optimization_procedure=optim_cifar10_resnet18,
                    baseline_type="single",
                    calibration_set=dm.get_val_set,
                    **vars(args),
                )
                for i in range(len(list_dm))
            ]

            cli_main(list_model, list_dm, root, "logs/dummy", args)

        with ArgvContext(
            "file.py",
            "--mixtype",
            "kernel_warping",
            "--mixup_alpha",
            "1.",
            "--dist_sim",
            "emb",
            "--val_temp_scaling",
            "--use_cv",
        ):
            args = init_args(
                DummyClassificationBaseline, DummyClassificationDataModule
            )

            args.root = str(root / "data")
            dm = DummyClassificationDataModule(num_classes=10, **vars(args))
            dm.dataset = (
                lambda root,
                num_channels,
                num_classes,
                image_size,
                transform,
                num_images: DummyClassificationDataset(
                    root,
                    num_channels=num_channels,
                    num_classes=num_classes,
                    image_size=image_size,
                    transform=transform,
                    num_images=20,
                )
            )

            list_dm = dm.make_cross_val_splits(2, 1)
            list_model = []
            for i in range(len(list_dm)):
                list_model.append(
                    DummyClassificationBaseline(
                        num_classes=list_dm[i].dm.num_classes,
                        in_channels=list_dm[i].dm.num_channels,
                        loss=nn.CrossEntropyLoss,
                        optimization_procedure=optim_cifar10_resnet18,
                        baseline_type="single",
                        calibration_set=dm.get_val_set,
                        **vars(args),
                    )
                )

            cli_main(list_model, list_dm, root, "logs/dummy", args)

    def test_classification_failures(self):
        with pytest.raises(ValueError):
            ClassificationSingle(
                10, nn.Module(), None, None, use_entropy=True, use_logits=True
            )

        with pytest.raises(ValueError):
            ClassificationSingle(10, nn.Module(), None, None, cutmix_alpha=-1)

        with pytest.raises(ValueError):
            ClassificationSingle(
                10, nn.Module(), None, None, eval_grouping_loss=True
            )

        model = dummy_model(1, 1, 1, 0, with_feats=False, with_linear=True)

        with pytest.raises(ValueError):
            ClassificationSingle(10, model, None, None, eval_grouping_loss=True)

        model = dummy_model(1, 1, 1, 0, with_feats=True, with_linear=False)

        with pytest.raises(ValueError):
            ClassificationSingle(10, model, None, None, eval_grouping_loss=True)


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

            cli_main(model, dm, root, "logs/dummy", args)

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

            cli_main(model, dm, root, "logs/dummy", args)

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

            cli_main(model, dm, root, "logs/dummy", args)

        with ArgvContext("file.py", "--eval-ood", "--entropy"):
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

            cli_main(model, dm, root, "logs/dummy", args)

        with ArgvContext("file.py", "--eval-ood", "--variation_ratio"):
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

            cli_main(model, dm, root, "logs/dummy", args)

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
