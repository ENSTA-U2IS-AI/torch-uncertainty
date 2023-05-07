# fmt:off
from argparse import ArgumentParser

from torch_uncertainty.datamodules import ImageNetDataModule


# fmt:on
class TestImageNetDataModule:
    """Testing the ImageNetDataModule datamodule class."""

    def test_imagenet(self):
        parser = ArgumentParser()
        parser = ImageNetDataModule.add_argparse_args(parser)

        args = parser.parse_args("")
        _ = ImageNetDataModule(**vars(args))
