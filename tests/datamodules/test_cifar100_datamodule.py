from argparse import ArgumentParser
from pathlib import Path

import pytest
from torchvision.datasets import CIFAR100

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules import CIFAR100DataModule
from torch_uncertainty.transforms import Cutout


class TestCIFAR100DataModule:
    """Testing the CIFAR100DataModule datamodule class."""

    def test_cifar100(self):
        parser = ArgumentParser()
        parser = CIFAR100DataModule.add_argparse_args(parser)

        # Simulate that cutout is set to 8
        args = parser.parse_args("")
        args.cutout = 8

        dm = CIFAR100DataModule(**vars(args))

        assert dm.dataset == CIFAR100
        assert isinstance(dm.train_transform.transforms[2], Cutout)

        dm.dataset = DummyClassificationDataset
        dm.ood_dataset = DummyClassificationDataset

        dm.prepare_data()
        dm.setup()
        dm.setup("test")

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()

        dm.eval_ood = True
        dm.prepare_data()
        dm.setup("test")
        dm.test_dataloader()

        args.test_alt = "c"
        args.cutout = 0
        args.root = Path(args.root)
        dm = CIFAR100DataModule(**vars(args))
        dm.dataset = DummyClassificationDataset
        with pytest.raises(ValueError):
            dm.setup()

        args.test_alt = None
        args.num_dataloaders = 2
        args.val_split = 0.1
        dm = CIFAR100DataModule(**vars(args))
        dm.dataset = DummyClassificationDataset
        dm.ood_dataset = DummyClassificationDataset
        dm.ood_dataset = DummyClassificationDataset
        dm.setup()
        dm.setup("test")
        dm.train_dataloader()
        with pytest.raises(ValueError):
            dm.setup("other")

        args.num_dataloaders = 1
        args.cutout = 8
        args.randaugment = True
        with pytest.raises(ValueError):
            dm = CIFAR100DataModule(**vars(args))

        args.cutout = None
        dm = CIFAR100DataModule(**vars(args))
        args.randaugment = False

        args.auto_augment = "rand-m9-n2-mstd0.5"
        dm = CIFAR100DataModule(**vars(args))

    def test_cifar100_cv(self):
        parser = ArgumentParser()
        parser = CIFAR100DataModule.add_argparse_args(parser)

        # Simulate that cutout is set to 8
        args = parser.parse_args("")

        dm = CIFAR100DataModule(**vars(args))
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
        dm = CIFAR100DataModule(**vars(args))
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
