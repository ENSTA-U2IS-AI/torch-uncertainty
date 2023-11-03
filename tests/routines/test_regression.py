from functools import partial
from pathlib import Path

import pytest
from cli_test_helpers import ArgvContext
from torch import nn

from tests._dummies import DummyRegressionBaseline, DummyRegressionDataModule
from torch_uncertainty import cli_main, init_args
from torch_uncertainty.losses import BetaNLL, NIGLoss
from torch_uncertainty.optimization_procedures import optim_cifar10_resnet18


class TestRegressionSingle:
    """Testing the Regression routine with a single model."""

    def test_cli_main_dummy_dist(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext("file.py"):
            args = init_args(DummyRegressionBaseline, DummyRegressionDataModule)

            # datamodule
            args.root = str(root / "data")
            dm = DummyRegressionDataModule(out_features=1, **vars(args))

            model = DummyRegressionBaseline(
                in_features=dm.in_features,
                out_features=2,
                loss=nn.GaussianNLLLoss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="single",
                dist_estimation=2,
                **vars(args),
            )

            cli_main(model, dm, root, "logs/dummy", args)

    def test_cli_main_dummy_dist_der(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext("file.py"):
            args = init_args(DummyRegressionBaseline, DummyRegressionDataModule)

            # datamodule
            args.root = str(root / "data")
            dm = DummyRegressionDataModule(out_features=1, **vars(args))

            loss = partial(
                NIGLoss,
                reg_weight=1e-2,
            )

            model = DummyRegressionBaseline(
                in_features=dm.in_features,
                out_features=4,
                loss=loss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="single",
                dist_estimation=4,
                **vars(args),
            )

            cli_main(model, dm, root, "logs/dummy_der", args)

    def test_cli_main_dummy_dist_betanll(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext("file.py"):
            args = init_args(DummyRegressionBaseline, DummyRegressionDataModule)

            # datamodule
            args.root = str(root / "data")
            dm = DummyRegressionDataModule(out_features=1, **vars(args))

            loss = partial(
                BetaNLL,
                beta=0.5,
            )

            model = DummyRegressionBaseline(
                in_features=dm.in_features,
                out_features=2,
                loss=loss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="single",
                dist_estimation=2,
                **vars(args),
            )

            cli_main(model, dm, root, "logs/dummy_betanll", args)

    def test_cli_main_dummy(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext("file.py"):
            args = init_args(DummyRegressionBaseline, DummyRegressionDataModule)

            # datamodule
            args.root = str(root / "data")
            dm = DummyRegressionDataModule(out_features=2, **vars(args))

            model = DummyRegressionBaseline(
                in_features=dm.in_features,
                out_features=dm.out_features,
                loss=nn.MSELoss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="single",
                **vars(args),
            )

            cli_main(model, dm, root, "logs/dummy", args)

    def test_regression_failures(self):
        with pytest.raises(ValueError):
            DummyRegressionBaseline(
                in_features=10,
                out_features=3,
                loss=nn.GaussianNLLLoss,
                optimization_procedure=optim_cifar10_resnet18,
                dist_estimation=4,
            )

        with pytest.raises(ValueError):
            DummyRegressionBaseline(
                in_features=10,
                out_features=3,
                loss=nn.GaussianNLLLoss,
                optimization_procedure=optim_cifar10_resnet18,
                dist_estimation=-4,
            )

        with pytest.raises(TypeError):
            DummyRegressionBaseline(
                in_features=10,
                out_features=4,
                loss=nn.GaussianNLLLoss,
                optimization_procedure=optim_cifar10_resnet18,
                dist_estimation=4.2,
            )


class TestRegressionEnsemble:
    """Testing the Regression routine with an ensemble model."""

    def test_cli_main_dummy(self):
        root = Path(__file__).parent.absolute().parents[0]
        with ArgvContext("file.py"):
            args = init_args(DummyRegressionBaseline, DummyRegressionDataModule)

            # datamodule
            args.root = str(root / "data")
            dm = DummyRegressionDataModule(out_features=1, **vars(args))

            model = DummyRegressionBaseline(
                in_features=dm.in_features,
                out_features=dm.out_features,
                loss=nn.MSELoss,
                optimization_procedure=optim_cifar10_resnet18,
                baseline_type="ensemble",
                **vars(args),
            )

            cli_main(model, dm, root, "logs/dummy", args)
