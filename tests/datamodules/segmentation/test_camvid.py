import pytest

from tests._dummies.dataset import DummySegmentationDataset
from torch_uncertainty.datamodules.segmentation import CamVidDataModule
from torch_uncertainty.datasets.segmentation import CamVid


class TestCamVidDataModule:
    """Testing the CamVidDataModule datamodule."""

    def test_camvid_main(self):
        dm = CamVidDataModule(
            root="./data/", batch_size=128, group_classes=False
        )
        dm = CamVidDataModule(
            root="./data/", batch_size=128, basic_augment=False
        )

        assert dm.dataset == CamVid

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
