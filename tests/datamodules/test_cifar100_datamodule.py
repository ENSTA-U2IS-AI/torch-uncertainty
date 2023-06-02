# fmt:off
from argparse import ArgumentParser

from torchvision.datasets import CIFAR100

from torch_uncertainty.datamodules import CIFAR100DataModule
from torch_uncertainty.transforms import Cutout

from .._dummies.dataset import DummyDataset


# fmt:on
class TestCIFAR100DataModule:
    """Testing the CIFAR100DataModule datamodule class."""

    def test_cifar100_cutout(self):
        parser = ArgumentParser()
        parser = CIFAR100DataModule.add_argparse_args(parser)

        # Simulate that cutout is set to 8
        args = parser.parse_args("")
        args.cutout = 8

        dm = CIFAR100DataModule(**vars(args))

        assert dm.dataset == CIFAR100
        assert isinstance(dm.transform_train.transforms[2], Cutout)

        dm.dataset = DummyDataset
        dm.prepare_data()
        dm.setup()
        dm.setup("test")

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()
