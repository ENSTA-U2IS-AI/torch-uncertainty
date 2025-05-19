from pathlib import Path

import pytest

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules.abstract import (
    CrossValDataModule,
    TUDataModule,
)


class TestTUDataModule:
    """Testing the TUDataModule class."""

    def test_errors(self) -> None:
        TUDataModule.__abstractmethods__ = set()
        dm = TUDataModule("root", 128, 128, 0.0, 4, True, True)
        dm.setup()
        with pytest.raises(NotImplementedError):
            dm._get_train_data()

        with pytest.raises(ValueError, match="The number of Test-time augmentations"):
            dm = TUDataModule(
                root="root",
                batch_size=128,
                eval_batch_size=128,
                val_split=0.0,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True,
                num_tta=75,
            )


class TestCrossValDataModule:
    """Testing the CrossValDataModule class."""

    def test_cv_main(self) -> None:
        TUDataModule.__abstractmethods__ = set()
        dm = TUDataModule("root", 128, 128, 0.0, 4, True, True)
        ds = DummyClassificationDataset(Path("root"))
        dm.train = ds
        dm.val = ds
        dm.test = ds
        cv_dm = CrossValDataModule("root", [0], [1], dm, 128, 128, 0.0, 4, True, True)

        cv_dm.setup()
        cv_dm.setup("test")

        # test abstract methods
        cv_dm.get_train_set()
        cv_dm.get_val_set()
        cv_dm.get_test_set()

        cv_dm.train_dataloader()
        cv_dm.val_dataloader()
        cv_dm.test_dataloader()

    def test_errors(self) -> None:
        TUDataModule.__abstractmethods__ = set()
        dm = TUDataModule(
            root="root",
            batch_size=128,
            eval_batch_size=None,
            val_split=0.0,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        ds = DummyClassificationDataset(Path("root"))
        dm.train = ds
        dm.val = ds
        dm.test = ds
        cv_dm = CrossValDataModule(
            root="root",
            train_idx=[0],
            val_idx=[1],
            datamodule=dm,
            batch_size=128,
            eval_batch_size=128,
            val_split=0.0,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
        )
        cv_dm.setup()
        with pytest.raises(NotImplementedError):
            cv_dm._get_train_data()

        with pytest.raises(ValueError):
            cv_dm.setup("other")
