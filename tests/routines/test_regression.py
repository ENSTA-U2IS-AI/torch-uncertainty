# fmt:off
from functools import partial
from pathlib import Path

from cli_test_helpers import ArgvContext
from torch import nn

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.losses import NIGLoss
from torch_uncertainty.optimization_procedures import optim_cifar10_resnet18

from .._dummies import DummyRegressionBaseline, DummyRegressionDataModule


# fmt:on
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

            cli_main(model, dm, root, "dummy", args)

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

            cli_main(model, dm, root, "dummy_der", args)

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

            cli_main(model, dm, root, "dummy", args)


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

            cli_main(model, dm, root, "dummy", args)
