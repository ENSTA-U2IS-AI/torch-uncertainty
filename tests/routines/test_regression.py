from pathlib import Path

import pytest
from lightning.pytorch import Trainer
from torch import nn

from tests._dummies import DummyRegressionBaseline, DummyRegressionDataModule
from torch_uncertainty.losses import DistributionNLL
from torch_uncertainty.optimization_procedures import optim_cifar10_resnet18
from torch_uncertainty.routines import RegressionRoutine


class TestRegression:
    """Testing the Regression routine."""

    def test_one_estimator_one_output(self):
        trainer = Trainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummyRegressionDataModule(out_features=1, root=root, batch_size=4)

        model = DummyRegressionBaseline(
            probabilistic=True,
            in_features=dm.in_features,
            num_outputs=1,
            loss=DistributionNLL,
            optimization_procedure=optim_cifar10_resnet18,
            baseline_type="single",
        )

        trainer.fit(model, dm)
        trainer.test(model, dm)

        model(dm.get_test_set()[0][0])

        model = DummyRegressionBaseline(
            probabilistic=False,
            in_features=dm.in_features,
            num_outputs=1,
            loss=DistributionNLL,
            optimization_procedure=optim_cifar10_resnet18,
            baseline_type="single",
        )

        trainer.fit(model, dm)
        trainer.test(model, dm)

    def test_one_estimator_two_outputs(self):
        trainer = Trainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummyRegressionDataModule(out_features=2, root=root, batch_size=4)

        model = DummyRegressionBaseline(
            probabilistic=True,
            in_features=dm.in_features,
            num_outputs=2,
            loss=DistributionNLL,
            optimization_procedure=optim_cifar10_resnet18,
            baseline_type="single",
        )

        trainer.fit(model, dm)
        trainer.test(model, dm)

        model = DummyRegressionBaseline(
            probabilistic=False,
            in_features=dm.in_features,
            num_outputs=2,
            loss=DistributionNLL,
            optimization_procedure=optim_cifar10_resnet18,
            baseline_type="single",
        )

        trainer.fit(model, dm)
        trainer.test(model, dm)

    def test_two_estimators_one_output(self):
        trainer = Trainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummyRegressionDataModule(out_features=1, root=root, batch_size=4)

        model = DummyRegressionBaseline(
            probabilistic=True,
            in_features=dm.in_features,
            num_outputs=1,
            loss=DistributionNLL,
            optimization_procedure=optim_cifar10_resnet18,
            baseline_type="ensemble",
        )

        trainer.fit(model, dm)
        trainer.test(model, dm)

        model = DummyRegressionBaseline(
            probabilistic=False,
            in_features=dm.in_features,
            num_outputs=1,
            loss=DistributionNLL,
            optimization_procedure=optim_cifar10_resnet18,
            baseline_type="ensemble",
        )

        trainer.fit(model, dm)
        trainer.test(model, dm)

    def test_two_estimators_two_outputs(self):
        trainer = Trainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummyRegressionDataModule(out_features=2, root=root, batch_size=4)

        model = DummyRegressionBaseline(
            probabilistic=True,
            in_features=dm.in_features,
            num_outputs=2,
            loss=DistributionNLL,
            optimization_procedure=optim_cifar10_resnet18,
            baseline_type="ensemble",
        )

        trainer.fit(model, dm)
        trainer.test(model, dm)

        model = DummyRegressionBaseline(
            probabilistic=False,
            in_features=dm.in_features,
            num_outputs=2,
            loss=DistributionNLL,
            optimization_procedure=optim_cifar10_resnet18,
            baseline_type="ensemble",
        )

        trainer.fit(model, dm)
        trainer.test(model, dm)

    def test_regression_failures(self):
        with pytest.raises(ValueError):
            RegressionRoutine(
                True, 1, nn.Identity(), nn.MSELoss, num_estimators=0
            )

        with pytest.raises(ValueError):
            RegressionRoutine(
                True, 0, nn.Identity(), nn.MSELoss, num_estimators=1
            )
