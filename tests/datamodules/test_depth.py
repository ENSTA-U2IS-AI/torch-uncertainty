import pytest

from tests._dummies.dataset import DummPixelRegressionDataset
from torch_uncertainty.datamodules.depth import (
    KITTIDataModule,
    MUADDataModule,
    NYUv2DataModule,
)
from torch_uncertainty.datasets import MUAD, KITTIDepth, NYUv2


class TestMUADDataModule:
    """Testing the MUADDataModule datamodule."""

    def test_muad_main(self):
        dm = MUADDataModule(
            root="./data/", min_depth=0, max_depth=100, batch_size=128
        )

        assert dm.dataset == MUAD

        dm.dataset = DummPixelRegressionDataset

        dm.prepare_data()
        dm.setup()

        with pytest.raises(ValueError):
            dm.setup("xxx")

        # test abstract methods
        dm.get_train_set()
        dm.get_val_set()
        dm.get_test_set()

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()

        dm.val_split = 0.1
        dm.prepare_data()
        dm.setup()
        dm.train_dataloader()
        dm.val_dataloader()


class TestNYUDataModule:
    """Testing the MUADDataModule datamodule."""

    def test_nyu_main(self):
        dm = NYUv2DataModule(root="./data/", max_depth=100, batch_size=128)

        assert dm.dataset == NYUv2

        dm.dataset = DummPixelRegressionDataset

        dm.prepare_data()
        dm.setup()

        with pytest.raises(ValueError):
            dm.setup("xxx")

        # test abstract methods
        dm.get_train_set()
        dm.get_val_set()
        dm.get_test_set()

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()

        dm.val_split = 0.1
        dm.prepare_data()
        dm.setup()
        dm.train_dataloader()
        dm.val_dataloader()

    def test_kitti_main(self):
        dm = KITTIDataModule(root="./data/", max_depth=100, batch_size=128)
        assert dm.dataset == KITTIDepth
