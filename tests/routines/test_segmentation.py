from pathlib import Path

import pytest
from lightning.pytorch import Trainer
from torch import nn

from tests._dummies import (
    DummySegmentationBaseline,
    DummySegmentationDataModule,
)
from torch_uncertainty.optim_recipes import optim_cifar10_resnet18
from torch_uncertainty.routines import SegmentationRoutine


class TestSegmentation:
    def test_one_estimator_two_classes(self):
        trainer = Trainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummySegmentationDataModule(root=root, batch_size=4, num_classes=2)

        model = DummySegmentationBaseline(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            image_size=dm.image_size,
            loss=nn.CrossEntropyLoss(),
            baseline_type="single",
            optim_recipe=optim_cifar10_resnet18,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_two_estimators_two_classes(self):
        trainer = Trainer(accelerator="cpu", fast_dev_run=True)

        root = Path(__file__).parent.absolute().parents[0] / "data"
        dm = DummySegmentationDataModule(root=root, batch_size=4, num_classes=2)

        model = DummySegmentationBaseline(
            in_channels=dm.num_channels,
            num_classes=dm.num_classes,
            image_size=dm.image_size,
            loss=nn.CrossEntropyLoss(),
            baseline_type="ensemble",
            optim_recipe=optim_cifar10_resnet18,
        )

        trainer.fit(model, dm)
        trainer.validate(model, dm)
        trainer.test(model, dm)
        model(dm.get_test_set()[0][0])

    def test_segmentation_failures(self):
        with pytest.raises(ValueError):
            SegmentationRoutine(
                2, nn.Identity(), nn.CrossEntropyLoss(), num_estimators=0
            )
        with pytest.raises(ValueError):
            SegmentationRoutine(1, nn.Identity(), nn.CrossEntropyLoss())
