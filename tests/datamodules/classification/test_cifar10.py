import pytest
from torchvision.datasets import CIFAR10

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.transforms import Cutout


class TestCIFAR10DataModule:
    """Testing the CIFAR10DataModule datamodule class."""

    def test_cifar10_main(self):
        dm = CIFAR10DataModule(root="./data/", batch_size=128, cutout=16)

        assert dm.dataset == CIFAR10
        assert isinstance(dm.train_transform.transforms[2], Cutout)

        dm.dataset = DummyClassificationDataset
        dm.ood_dataset = DummyClassificationDataset
        dm.shift_dataset = DummyClassificationDataset

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

        dm.eval_ood = True
        dm.eval_shift = True
        dm.prepare_data()
        dm.setup("test")
        dm.test_dataloader()

        dm = CIFAR10DataModule(
            root="./data/",
            batch_size=128,
            cutout=16,
            test_alt="h",
            basic_augment=False,
        )
        dm.dataset = DummyClassificationDataset
        dm.setup("test")

        dm = CIFAR10DataModule(
            root="./data/",
            batch_size=128,
            cutout=16,
            num_dataloaders=2,
            val_split=0.1,
        )
        dm.dataset = DummyClassificationDataset
        dm.ood_dataset = DummyClassificationDataset
        dm.setup()
        dm.train_dataloader()

        with pytest.raises(ValueError):
            dm = CIFAR10DataModule(
                root="./data/",
                batch_size=128,
                cutout=8,
                num_dataloaders=2,
                val_split=0.1,
                auto_augment="rand-m9-n2-mstd0.5",
            )

        with pytest.raises(
            ValueError, match="CIFAR-H can only be used in testing."
        ):
            dm = CIFAR10DataModule(
                root="./data/",
                batch_size=128,
                test_alt="h",
            )
            dm.setup("fit")

        with pytest.raises(ValueError, match="Test set "):
            dm = CIFAR10DataModule(
                root="./data/",
                batch_size=128,
                test_alt="x",
            )

        dm = CIFAR10DataModule(
            root="./data/",
            batch_size=128,
            cutout=None,
            num_dataloaders=2,
            val_split=0.1,
            auto_augment="rand-m9-n2-mstd0.5",
        )

    def test_cifar10_cv(self):
        dm = CIFAR10DataModule(root="./data/", batch_size=128)
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

        dm = CIFAR10DataModule(root="./data/", batch_size=128, val_split=0.1)
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
