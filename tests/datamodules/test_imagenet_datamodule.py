# fmt:off
from argparse import ArgumentParser

from torchvision.datasets import ImageNet

from torch_uncertainty.datamodules import ImageNetDataModule

from ..datasets.dummy_dataset import DummyDataset


# fmt:on
class TestImageNetDataModule:
    """Testing the ImageNetDataModule datamodule class."""

    def test_imagenet(self):
        parser = ArgumentParser()
        parser = ImageNetDataModule.add_argparse_args(parser)

        args = parser.parse_args("")
        dm = ImageNetDataModule(**vars(args))

        assert dm.dataset == ImageNet

        dm.dataset = DummyDataset
        dm.prepare_data()
        dm.setup()
        dm.setup("test")

        dm.train_dataloader()
        dm.val_dataloader()
        dm.test_dataloader()
