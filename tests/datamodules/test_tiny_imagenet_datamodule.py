from argparse import ArgumentParser

import pytest

from tests._dummies.dataset import DummyClassificationDataset
from torch_uncertainty.datamodules import TinyImageNetDataModule
from torch_uncertainty.datasets.classification import TinyImageNet


class TestTinyImageNetDataModule:
    """Testing the TinyImageNetDataModule datamodule class."""

    def test_tiny_imagenet(self):
        parser = ArgumentParser()
        parser = TinyImageNetDataModule.add_argparse_args(parser)

        args = parser.parse_args("")
        dm = TinyImageNetDataModule(**vars(args))

        assert dm.dataset == TinyImageNet

        args.rand_augment_opt = "rand-m9-n3-mstd0.5"
        args.ood_ds = "imagenet-o"
        dm = TinyImageNetDataModule(**vars(args))

        args.ood_ds = "textures"
        dm = TinyImageNetDataModule(**vars(args))

        args.ood_ds = "other"
        with pytest.raises(ValueError):
            TinyImageNetDataModule(**vars(args))

        dm.dataset = DummyClassificationDataset
        dm.ood_dataset = DummyClassificationDataset

        dm.prepare_data()
        dm.setup()
        dm.setup("test")

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()

        with pytest.raises(ValueError):
            dm.setup("other")

        dm.eval_ood = True
        dm.prepare_data()
        dm.setup("test")
        dm.test_dataloader()

    def test_tiny_imagenet_cv(self):
        parser = ArgumentParser()
        parser = TinyImageNetDataModule.add_argparse_args(parser)

        # Simulate that cutout is set to 8
        args = parser.parse_args("")

        dm = TinyImageNetDataModule(**vars(args))
        dm.dataset = lambda root, split, transform: DummyClassificationDataset(
            root, split=split, transform=transform, num_images=20
        )
        dm.make_cross_val_splits(2, 1)

        args.val_split = 0.1
        dm = TinyImageNetDataModule(**vars(args))
        dm.dataset = lambda root, split, transform: DummyClassificationDataset(
            root, split=split, transform=transform, num_images=20
        )
        dm.make_cross_val_splits(2, 1)
