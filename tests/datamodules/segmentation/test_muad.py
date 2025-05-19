import pytest
from torch import nn

from tests._dummies.dataset import DummySegmentationDataset
from torch_uncertainty.datamodules.segmentation import MUADDataModule
from torch_uncertainty.datasets import MUAD


class TestMUADDataModule:
    """Testing the MUADDataModule datamodule."""

    def test_muad_main(self) -> None:
        dm = MUADDataModule(
            root="./data/",
            batch_size=128,
            train_transform=nn.Identity(),
            test_transform=nn.Identity(),
            version="small",
        )
        assert isinstance(dm.train_transform, nn.Identity)
        assert isinstance(dm.test_transform, nn.Identity)
        dm = MUADDataModule(root="./data/", batch_size=128, eval_ood=True)

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

        dm.eval_ood = False
        dm.val_split = 0.1
        dm.prepare_data()
        dm.setup()
        dm.train_dataloader()
        dm.val_dataloader()
