from pathlib import Path

import pytest
from torch import nn

from tests._dummies import (
    DummyDepthBaseline,
    DummyDepthDataModule,
)
from torch_uncertainty.optim_recipes import optim_cifar10_resnet18
from torch_uncertainty.routines import DepthRoutine
from torch_uncertainty.utils import TUTrainer


class TestDepth:
    def test_one_estimator_two_classes(self):
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummyDepthDataModule(root=root, batch_size=4, output_dim=2)

        model = DummyDepthBaseline(
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

    def test_two_estimators_two_classes(self):
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummyDepthDataModule(root=root, batch_size=4, output_dim=2)

        model = DummyDepthBaseline(
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
        with pytest.raises(
            ValueError, match="num_estimators must be positive, got"
        ):
            DepthRoutine(
                model=nn.Identity(),
                output_dim=2,
                loss=nn.MSELoss(),
                num_estimators=0,
                probabilistic=False,
            )
