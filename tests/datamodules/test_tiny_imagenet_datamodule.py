# fmt:off
from argparse import ArgumentParser

import pytest

from torch_uncertainty.datamodules import TinyImageNetDataModule
from torch_uncertainty.datasets.classification import TinyImageNet

from .._dummies.dataset import DummyClassificationDataset



class TestTinyImageNetDataModule:
    """Testing the TinyImageNetDataModule datamodule class."""

    def test_imagenet(self):
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

        dm.ood_detection = True
        dm.prepare_data()
        dm.setup("test")
        dm.test_dataloader()
