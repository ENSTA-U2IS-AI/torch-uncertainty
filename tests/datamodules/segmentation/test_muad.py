import pytest

from tests._dummies.dataset import DummySegmentationDataset
from torch_uncertainty.datamodules.segmentation import MUADDataModule
from torch_uncertainty.datasets import MUAD


class TestMUADDataModule:
    """Testing the MUADDataModule datamodule."""

    def test_camvid_main(self):
        dm = MUADDataModule(root="./data/", batch_size=128)

        assert dm.dataset == MUAD

        dm.dataset = DummySegmentationDataset

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
