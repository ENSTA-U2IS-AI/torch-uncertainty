# fmt:off
from argparse import ArgumentParser

import pytest
from torch import nn
from torchvision.datasets import MNIST

from torch_uncertainty.datamodules import MNISTDataModule
from torch_uncertainty.transforms import Cutout

from .._dummies.dataset import DummyClassificationDataset



class TestMNISTDataModule:
    """Testing the MNISTDataModule datamodule class."""

    def test_mnist_cutout(self):
        parser = ArgumentParser()
        parser = MNISTDataModule.add_argparse_args(parser)

        # Simulate that cutout is set to 16
        args = parser.parse_args("")
        args.cutout = 16
        args.val_split = 0.1
        dm = MNISTDataModule(**vars(args))

        assert dm.dataset == MNIST
        assert isinstance(dm.transform_train.transforms[0], Cutout)

        args.root = str(args.root)
        args.ood_ds = "not"
        args.cutout = 0
        args.val_split = 0
        dm = MNISTDataModule(**vars(args))
        assert isinstance(dm.transform_train.transforms[0], nn.Identity)

        args.ood_ds = "other"
        with pytest.raises(ValueError):
            MNISTDataModule(**vars(args))

        dm.dataset = DummyClassificationDataset
        dm.ood_dataset = DummyClassificationDataset

        dm.prepare_data()
        dm.setup()
        dm.setup("test")

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()

        dm.evaluate_ood = True
        dm.val_split = 0.1
        dm.prepare_data()
        dm.setup("test")
        dm.test_dataloader()
