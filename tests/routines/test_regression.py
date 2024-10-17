from pathlib import Path

import pytest
from torch import nn

from tests._dummies import DummyRegressionBaseline, DummyRegressionDataModule
from torch_uncertainty import TUTrainer
from torch_uncertainty.losses import DistributionNLLLoss
from torch_uncertainty.optim_recipes import optim_cifar10_resnet18
from torch_uncertainty.routines import RegressionRoutine


class TestRegression:
    """Testing the Regression routine."""

    def test_one_estimator_one_output(self):
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummyRegressionDataModule(out_features=1, root=root, batch_size=4)

        model = DummyRegressionBaseline(
            probabilistic=True,
            in_features=dm.in_features,
            output_dim=1,
            loss=DistributionNLLLoss(),
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="single",
            ema=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)
        model = DummyRegressionBaseline(
            probabilistic=False,
            in_features=dm.in_features,
            output_dim=1,
            loss=nn.MSELoss(),
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="single",
            swa=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_one_estimator_two_outputs(self):
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummyRegressionDataModule(out_features=2, root=root, batch_size=4)

        model = DummyRegressionBaseline(
            probabilistic=True,
            in_features=dm.in_features,
            output_dim=2,
            loss=DistributionNLLLoss(),
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="single",
            dist_type="laplace",
        )
        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)
        model = DummyRegressionBaseline(
            probabilistic=False,
            in_features=dm.in_features,
            output_dim=2,
            loss=nn.MSELoss(),
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="single",
        )
        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_two_estimators_one_output(self):
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummyRegressionDataModule(out_features=1, root=root, batch_size=4)

        model = DummyRegressionBaseline(
            probabilistic=True,
            in_features=dm.in_features,
            output_dim=1,
            loss=DistributionNLLLoss(),
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="ensemble",
            dist_type="nig",
        )
        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)
        model = DummyRegressionBaseline(
            probabilistic=False,
            in_features=dm.in_features,
            output_dim=1,
            loss=nn.MSELoss(),
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="ensemble",
        )
        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_two_estimators_two_outputs(self):
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummyRegressionDataModule(out_features=2, root=root, batch_size=4)

        model = DummyRegressionBaseline(
            probabilistic=True,
            in_features=dm.in_features,
            output_dim=2,
            loss=DistributionNLLLoss(),
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="ensemble",
        )
        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)
        model = DummyRegressionBaseline(
            probabilistic=False,
            in_features=dm.in_features,
            output_dim=2,
            loss=nn.MSELoss(),
            optim_recipe=optim_cifar10_resnet18,
            baseline_type="ensemble",
        )
        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_regression_failures(self):
        with pytest.raises(ValueError, match="output_dim must be positive"):
            RegressionRoutine(
                probabilistic=True,
                output_dim=0,
                model=nn.Identity(),
                loss=nn.MSELoss(),
            )
