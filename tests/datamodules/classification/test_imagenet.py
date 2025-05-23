from pathlib import Path

import pytest
from torch import nn

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules import ImageNetDataModule


class TestImageNetDataModule:
    """Testing the ImageNetDataModule datamodule class."""

    def test_imagenet(self):
        dm = ImageNetDataModule(
            root="./data/",
            batch_size=128,
            val_split=0.1,
            train_transform=nn.Identity(),
            test_transform=nn.Identity(),
        )
        assert isinstance(dm.train_transform, nn.Identity)
        assert isinstance(dm.test_transform, nn.Identity)
        dm.dataset = DummyClassificationDataset

        dm.prepare_data()
        dm.setup()

        path = Path(__file__).parent.resolve() / "../../assets/dummy_indices.yaml"
        dm = ImageNetDataModule(root="./data/", batch_size=128, val_split=path)
        dm.shift_dataset = DummyClassificationDataset
        dm.setup("fit")
        dm.setup("test")
        dm.val_dataloader()
        dm.test_dataloader()

        dm.val_split = None
        dm.setup("fit")
        dm.val_dataloader()
        dm.test_dataloader()

        dm.eval_ood = True
        dm.eval_shift = True
        dm.prepare_data()
        dm.setup("test")
        dm.test_dataloader()

        ImageNetDataModule(
            root="./data/",
            batch_size=128,
            val_split=path,
            rand_augment_opt="rand-m9-n1",
            basic_augment=False,
        )

        with pytest.raises(ValueError):
            dm.setup("other")

        for test_alt in ["r", "o", "a"]:
            dm = ImageNetDataModule(root="./data/", batch_size=128, test_alt=test_alt)

        with pytest.raises(ValueError):
            dm = ImageNetDataModule(root="./data/", batch_size=128, test_alt="x")

        for procedure in ["ViT", "A3"]:
            dm = ImageNetDataModule(
                root="./data/",
                batch_size=128,
                procedure=procedure,
            )

        with pytest.raises(ValueError):
            dm = ImageNetDataModule(root="./data/", batch_size=128, procedure="A2")

        with pytest.raises(FileNotFoundError):
            dm._verify_splits(split="test")

        dm.root = Path("./tests/testlog")
        with pytest.raises(FileNotFoundError):
            dm._verify_splits(split="test")
