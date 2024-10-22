from pathlib import Path

import pytest
from torchvision.datasets import ImageNet

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules import ImageNetDataModule


class TestImageNetDataModule:
    """Testing the ImageNetDataModule datamodule class."""

    def test_imagenet(self):
        dm = ImageNetDataModule(root="./data/", batch_size=128, val_split=0.1)
        assert dm.dataset == ImageNet
        dm.dataset = DummyClassificationDataset
        dm.ood_dataset = DummyClassificationDataset
        dm.prepare_data()
        dm.setup()

        path = (
            Path(__file__).parent.resolve() / "../../assets/dummy_indices.yaml"
        )
        dm = ImageNetDataModule(root="./data/", batch_size=128, val_split=path)
        dm.dataset = DummyClassificationDataset
        dm.ood_dataset = DummyClassificationDataset
        dm.shift_dataset = DummyClassificationDataset
        dm.setup("fit")
        dm.setup("test")
        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()

        dm.val_split = None
        dm.setup("fit")
        dm.train_dataloader()
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
            dm = ImageNetDataModule(
                root="./data/", batch_size=128, test_alt=test_alt
            )

        with pytest.raises(ValueError):
            dm.setup()

        with pytest.raises(ValueError):
            dm = ImageNetDataModule(
                root="./data/", batch_size=128, test_alt="x"
            )

        for ood_ds in ["inaturalist", "imagenet-o", "textures", "openimage-o"]:
            dm = ImageNetDataModule(
                root="./data/", batch_size=128, ood_ds=ood_ds
            )
            if ood_ds == "inaturalist":
                dm.eval_ood = True
                dm.dataset = DummyClassificationDataset
                dm.ood_dataset = DummyClassificationDataset
                dm.prepare_data()
                dm.setup("test")
                dm.test_dataloader()

        with pytest.raises(ValueError):
            dm = ImageNetDataModule(
                root="./data/", batch_size=128, ood_ds="other"
            )

        for procedure in ["ViT", "A3"]:
            dm = ImageNetDataModule(
                root="./data/",
                batch_size=128,
                ood_ds="svhn",
                procedure=procedure,
            )

        with pytest.raises(ValueError):
            dm = ImageNetDataModule(
                root="./data/", batch_size=128, procedure="A2"
            )

        with pytest.raises(FileNotFoundError):
            dm._verify_splits(split="test")

        with pytest.raises(FileNotFoundError):
            dm.root = Path("./tests/testlog")
            dm._verify_splits(split="test")
