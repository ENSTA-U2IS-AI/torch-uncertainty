import pytest

from torch_uncertainty.datamodules.abstract import (
    AbstractDataModule,
    CrossValDataModule,
)

from .._dummies.dataset import DummyClassificationDataset


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

    def test_errors(self):
        dm = AbstractDataModule("root", 128, 4, True, True)
        ds = DummyClassificationDataset("root")
        dm.train = ds
        dm.val = ds
        dm.test = ds
        cv_dm = CrossValDataModule("root", [0], [1], dm, 128, 4, True, True)
        with pytest.raises(NotImplementedError):
            cv_dm.setup()
            cv_dm._get_train_data()
            cv_dm._get_train_targets()
