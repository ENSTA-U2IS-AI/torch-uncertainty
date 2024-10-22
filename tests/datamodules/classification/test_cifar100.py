import pytest
from torchvision.datasets import CIFAR100

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules import CIFAR100DataModule
from torch_uncertainty.transforms import Cutout


class TestCIFAR100DataModule:
    """Testing the CIFAR100DataModule datamodule class."""

    def test_cifar100(self):
        dm = CIFAR100DataModule(root="./data/", batch_size=128, cutout=16)

        assert dm.dataset == CIFAR100
        assert isinstance(dm.train_transform.transforms[2], Cutout)

        dm.dataset = DummyClassificationDataset
        dm.ood_dataset = DummyClassificationDataset
        dm.shift_dataset = DummyClassificationDataset

        dm.prepare_data()
        dm.setup()

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()

        dm.eval_ood = True
        dm.eval_shift = True
        dm.prepare_data()
        dm.setup("test")
        dm.test_dataloader()

        dm = CIFAR100DataModule(
            root="./data/",
            batch_size=128,
            cutout=0,
            val_split=0.1,
            num_dataloaders=2,
            basic_augment=False,
        )
        dm.dataset = DummyClassificationDataset
        dm.ood_dataset = DummyClassificationDataset
        dm.shift_dataset = DummyClassificationDataset
        dm.setup()
        dm.setup("test")
        dm.train_dataloader()
        with pytest.raises(ValueError):
            dm.setup("other")

        with pytest.raises(ValueError):
            dm = CIFAR100DataModule(
                root="./data/",
                batch_size=128,
                num_dataloaders=1,
                cutout=8,
                randaugment=True,
            )

        dm = CIFAR100DataModule(
            root="./data/", batch_size=128, randaugment=True
        )

        dm = CIFAR100DataModule(
            root="./data/", batch_size=128, auto_augment="rand-m9-n2-mstd0.5"
        )

    def test_cifar100_cv(self):
        dm = CIFAR100DataModule(root="./data/", batch_size=128)
        dm.dataset = (
            lambda root, train, download, transform: DummyClassificationDataset(
                root,
                train=train,
                download=download,
                transform=transform,
                num_images=20,
            )
        )
        dm.make_cross_val_splits(2, 1)

        dm = CIFAR100DataModule(root="./data/", batch_size=128, val_split=0.1)
        dm.dataset = (
            lambda root, train, download, transform: DummyClassificationDataset(
                root,
                train=train,
                download=download,
                transform=transform,
                num_images=20,
            )
        )
        dm.make_cross_val_splits(2, 1)
