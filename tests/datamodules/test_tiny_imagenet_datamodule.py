# fmt:off
from argparse import ArgumentParser

from torch_uncertainty.datamodules import TinyImageNetDataModule
from torch_uncertainty.datasets import TinyImageNet

from .._dummies.dataset import DummyClassificationDataset


# fmt:on
class TestTinyImageNetDataModule:
    """Testing the TinyImageNetDataModule datamodule class."""

    def test_imagenet(self):
        parser = ArgumentParser()
        parser = TinyImageNetDataModule.add_argparse_args(parser)

        args = parser.parse_args("")
        dm = TinyImageNetDataModule(**vars(args))

        assert dm.dataset == TinyImageNet

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
