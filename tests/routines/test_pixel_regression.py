from pathlib import Path

import pytest
from torch import nn

from tests._dummies import (
    DummPixelRegressionDataModule,
    DummyPixelRegressionBaseline,
)
from torch_uncertainty.optim_recipes import optim_cifar10_resnet18
from torch_uncertainty.routines import PixelRegressionRoutine
from torch_uncertainty.utils import TUTrainer


class TestDepth:
    def test_one_estimator_two_classes(self):
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummPixelRegressionDataModule(
            root=root, batch_size=4, output_dim=2
        )

        model = DummyPixelRegressionBaseline(
            in_channels=dm.num_channels,
            output_dim=dm.output_dim,
            image_size=dm.image_size,
            loss=nn.MSELoss(),
            baseline_type="single",
            optim_recipe=optim_cifar10_resnet18,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_two_estimators_one_class(self):
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummPixelRegressionDataModule(
            root=root, batch_size=4, output_dim=1
        )

        model = DummyPixelRegressionBaseline(
            in_channels=dm.num_channels,
            output_dim=dm.output_dim,
            image_size=dm.image_size,
            loss=nn.MSELoss(),
            baseline_type="ensemble",
            optim_recipe=optim_cifar10_resnet18,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_depth_errors(self):
        with pytest.raises(ValueError, match="output_dim must be positive"):
            PixelRegressionRoutine(
                model=nn.Identity(),
                output_dim=0,
                loss=nn.MSELoss(),
                probabilistic=False,
            )
