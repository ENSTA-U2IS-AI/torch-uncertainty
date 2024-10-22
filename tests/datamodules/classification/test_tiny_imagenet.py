import pytest

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules import TinyImageNetDataModule
from torch_uncertainty.datasets.classification import TinyImageNet


class TestTinyImageNetDataModule:
    """Testing the TinyImageNetDataModule datamodule class."""

    def test_tiny_imagenet(self):
        dm = TinyImageNetDataModule(root="./data/", batch_size=128)

        assert dm.dataset == TinyImageNet

        dm = TinyImageNetDataModule(
            root="./data/",
            batch_size=128,
            rand_augment_opt="rand-m9-n3-mstd0.5",
            ood_ds="imagenet-o",
        )

        dm = TinyImageNetDataModule(
            root="./data/",
            batch_size=128,
            ood_ds="textures",
            basic_augment=False,
        )

        with pytest.raises(ValueError):
            TinyImageNetDataModule(
                root="./data/", batch_size=128, ood_ds="other"
            )

        dm.dataset = DummyClassificationDataset
        dm.ood_dataset = DummyClassificationDataset
        dm.shift_dataset = DummyClassificationDataset

        dm.prepare_data()
        dm.setup()

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()

        with pytest.raises(ValueError):
            dm.setup("other")

        dm.eval_ood = True
        dm.eval_shift = True
        dm.prepare_data()
        dm.setup("test")
        dm.test_dataloader()

        dm = TinyImageNetDataModule(
            root="./data/", batch_size=128, ood_ds="svhn"
        )
        dm.dataset = DummyClassificationDataset
        dm.ood_dataset = DummyClassificationDataset
        dm.shift_dataset = DummyClassificationDataset
        dm.eval_ood = True
        dm.eval_shift = True
        dm.prepare_data()
        dm.setup("test")

    def test_tiny_imagenet_cv(self):
        dm = TinyImageNetDataModule(root="./data/", batch_size=128)
        dm.dataset = lambda root, split, transform: DummyClassificationDataset(
            root, split=split, transform=transform, num_images=20
        )
        dm.make_cross_val_splits(2, 1)

        dm = TinyImageNetDataModule(
            root="./data/", batch_size=128, val_split=0.1
        )
        dm.dataset = lambda root, split, transform: DummyClassificationDataset(
            root, split=split, transform=transform, num_images=20
        )
        dm.make_cross_val_splits(2, 1)
