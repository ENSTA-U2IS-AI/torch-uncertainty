from pathlib import Path

import pytest
from torch import nn

from tests._dummies import (
    DummySegmentationBaseline,
    DummySegmentationDataModule,
)
from torch_uncertainty.optim_recipes import optim_cifar10_resnet18
from torch_uncertainty.routines import SegmentationRoutine
from torch_uncertainty.utils import TUTrainer


class TestSegmentation:
    def test_one_estimator_two_classes(self):
        trainer = TUTrainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummySegmentationDataModule(root=root, batch_size=4, num_classes=2)

        model = DummySegmentationBaseline(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            image_size=dm.image_size,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            optim_recipe=optim_cifar10_resnet18,
            log_plots=True,
            ema=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

        trainer = TUTrainer(
            accelerator="cpu",
            max_epochs=2,
            logger=None,
            enable_checkpointing=False,
        )

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummySegmentationDataModule(root=root, batch_size=4, num_classes=2)

        model = DummySegmentationBaseline(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            image_size=dm.image_size,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            optim_recipe=optim_cifar10_resnet18,
            log_plots=True,
            swa=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_two_estimators_two_classes(self):
        trainer = TUTrainer(
            accelerator="cpu",
            max_epochs=2,
            logger=None,
            enable_checkpointing=False,
        )

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummySegmentationDataModule(root=root, batch_size=4, num_classes=2)

        model = DummySegmentationBaseline(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            image_size=dm.image_size,
            loss=nn.CrossEntropyLoss(),
            baseline_type="ensemble",
            optim_recipe=optim_cifar10_resnet18,
            swa=True,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_segmentation_errors(self):
        with pytest.raises(
            ValueError, match="num_classes must be at least 2, got"
        ):
            SegmentationRoutine(
                model=nn.Identity(), num_classes=1, loss=nn.CrossEntropyLoss()
            )

        with pytest.raises(
            ValueError, match="metric_subsampling_rate must be in"
        ):
            SegmentationRoutine(
                model=nn.Identity(),
                num_classes=2,
                loss=nn.CrossEntropyLoss(),
                metric_subsampling_rate=-1,
            )

        with pytest.raises(
            ValueError, match="num_calibration_bins must be at least 2, got"
        ):
            SegmentationRoutine(
                model=nn.Identity(),
                num_classes=2,
                loss=nn.CrossEntropyLoss(),
                num_calibration_bins=0,
            )
