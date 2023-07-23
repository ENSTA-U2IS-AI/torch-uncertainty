# fmt:off
from argparse import ArgumentParser

from torchvision.datasets import CIFAR10

from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.transforms import Cutout

from .._dummies.dataset import DummyClassificationDataset


# fmt:on
class TestCIFAR10DataModule:
    """Testing the CIFAR10DataModule datamodule class."""

    def test_cifar10_cutout(self):
        parser = ArgumentParser()
        parser = CIFAR10DataModule.add_argparse_args(parser)

        # Simulate that cutout is set to 8
        args = parser.parse_args("")
        args.cutout = 16

        dm = CIFAR10DataModule(**vars(args))

        assert dm.dataset == CIFAR10
        assert isinstance(dm.transform_train.transforms[2], Cutout)

        dm.dataset = DummyClassificationDataset
        dm.prepare_data()
        dm.setup()
        dm.setup("test")

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()
