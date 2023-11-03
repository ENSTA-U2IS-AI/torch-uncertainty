from pathlib import Path

import pytest

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules.abstract import (
    AbstractDataModule,
    CrossValDataModule,
)


class TestAbstractDataModule:
    """Testing the AbstractDataModule class."""

    def test_errors(self):
        dm = AbstractDataModule("root", 128, 4, True, True)
        with pytest.raises(NotImplementedError):
            dm.setup()
            dm._get_train_data()
            dm._get_train_targets()


class TestCrossValDataModule:
    """Testing the CrossValDataModule class."""

    def test_cv_main(self):
        dm = AbstractDataModule("root", 128, 4, True, True)
        ds = DummyClassificationDataset(Path("root"))
        dm.train = ds
        dm.val = ds
        dm.test = ds
        cv_dm = CrossValDataModule("root", [0], [1], dm, 128, 4, True, True)

        cv_dm.setup()
        cv_dm.setup("test")

        # test abstract methods
        cv_dm.get_train_set()
        cv_dm.get_val_set()
        cv_dm.get_test_set()

        cv_dm.train_dataloader()
        cv_dm.val_dataloader()
        cv_dm.test_dataloader()

    def test_errors(self):
        dm = AbstractDataModule("root", 128, 4, True, True)
        ds = DummyClassificationDataset(Path("root"))
        dm.train = ds
        dm.val = ds
        dm.test = ds
        cv_dm = CrossValDataModule("root", [0], [1], dm, 128, 4, True, True)
        with pytest.raises(NotImplementedError):
            cv_dm.setup()
            cv_dm._get_train_data()
            cv_dm._get_train_targets()
