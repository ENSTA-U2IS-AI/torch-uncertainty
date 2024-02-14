from argparse import ArgumentParser

import pytest
from torchvision.datasets import CIFAR10

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.transforms import Cutout


class TestCIFAR10DataModule:
    """Testing the CIFAR10DataModule datamodule class."""

    def test_cifar10_main(self):
        parser = ArgumentParser()
        parser = CIFAR10DataModule.add_argparse_args(parser)

        # Simulate that cutout is set to 8
        args = parser.parse_args("")
        args.cutout = 16

        dm = CIFAR10DataModule(**vars(args))

        assert dm.dataset == CIFAR10
        assert isinstance(dm.train_transform.transforms[2], Cutout)

        dm.dataset = DummyClassificationDataset
        dm.ood_dataset = DummyClassificationDataset

        dm.prepare_data()
        dm.setup()
        dm.setup("test")

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
        dm.prepare_data()
        dm.setup("test")
        dm.test_dataloader()

        args.test_alt = "c"
        dm = CIFAR10DataModule(**vars(args))
        dm.dataset = DummyClassificationDataset
        with pytest.raises(ValueError):
            dm.setup()

        args.test_alt = "h"
        dm = CIFAR10DataModule(**vars(args))
        dm.dataset = DummyClassificationDataset
        dm.setup("test")

        args.test_alt = None
        args.num_dataloaders = 2
        args.val_split = 0.1
        dm = CIFAR10DataModule(**vars(args))
        dm.dataset = DummyClassificationDataset
        dm.ood_dataset = DummyClassificationDataset
        dm.setup()
        dm.setup("test")
        dm.train_dataloader()

        args.cutout = 8
        args.auto_augment = "rand-m9-n2-mstd0.5"
        with pytest.raises(ValueError):
            dm = CIFAR10DataModule(**vars(args))

        args.cutout = None
        args.auto_augment = "rand-m9-n2-mstd0.5"
        dm = CIFAR10DataModule(**vars(args))

    def test_cifar10_cv(self):
        parser = ArgumentParser()
        parser = CIFAR10DataModule.add_argparse_args(parser)

        # Simulate that cutout is set to 8
        args = parser.parse_args("")

        dm = CIFAR10DataModule(**vars(args))
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

        args.val_split = 0.1
        dm = CIFAR10DataModule(**vars(args))
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
