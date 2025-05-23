import pytest
from torch import nn

from tests._dummies.dataset import DummPixelRegressionDataset
from torch_uncertainty.datamodules.depth import (
    KITTIDataModule,
    MUADDataModule,
    NYUv2DataModule,
)
from torch_uncertainty.datamodules.depth.base import DepthDataModule
from torch_uncertainty.datasets import MUAD, KITTIDepth, NYUv2


class TestMUADDataModule:
    """Testing the MUADDataModule datamodule."""

    def test_depth_dm(self) -> None:
        dm = DepthDataModule(
            dataset=DummPixelRegressionDataset,
            root="./data/",
            batch_size=128,
            min_depth=0,
            max_depth=100,
            train_transform=nn.Identity(),
            test_transform=nn.Identity(),
        )
        assert isinstance(dm.train_transform, nn.Identity)
        assert isinstance(dm.test_transform, nn.Identity)

    def test_depth_dm_failures(self) -> None:
        with pytest.raises(ValueError):
            DepthDataModule(
                dataset=DummPixelRegressionDataset,
                root="./data/",
                batch_size=128,
                min_depth=0,
                max_depth=100,
                eval_size=(224, 224),
            )

        with pytest.raises(ValueError):
            DepthDataModule(
                dataset=DummPixelRegressionDataset,
                root="./data/",
                batch_size=128,
                min_depth=0,
                max_depth=100,
                crop_size=(224, 224),
            )

    def test_muad_main(self) -> None:
        dm = MUADDataModule(root="./data/", min_depth=0, max_depth=100, batch_size=128)

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
    """Testing the NYUv2DataModule datamodule."""

    def test_nyu_main(self) -> None:
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

    def test_kitti_main(self) -> None:
        dm = KITTIDataModule(root="./data/", max_depth=100, batch_size=128)
        assert dm.dataset == KITTIDepth
