# fmt:off
from argparse import ArgumentParser

from torchvision.datasets import MNIST

from torch_uncertainty.datamodules import MNISTDataModule
from torch_uncertainty.transforms import Cutout

from .._dummies.dataset import DummyDataset


# fmt:on
class TestMNISTDataModule:
    """Testing the MNISTDataModule datamodule class."""

    def test_mnist_cutout(self):
        parser = ArgumentParser()
        parser = MNISTDataModule.add_argparse_args(parser)

        # Simulate that cutout is set to 16
        args = parser.parse_args("")
        args.cutout = 16

        dm = MNISTDataModule(**vars(args))

        assert dm.dataset == MNIST
        assert isinstance(dm.transform_train.transforms[0], Cutout)

        dm.dataset = DummyDataset
        dm.prepare_data()
        dm.setup()
        dm.setup("test")

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()
