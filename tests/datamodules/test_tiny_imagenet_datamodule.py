# fmt:off
from argparse import ArgumentParser

from torch_uncertainty.datamodules import TinyImageNetDataModule


# fmt:on
class TestTinyImageNetDataModule:
    """Testing the TinyImageNetDataModule datamodule class."""

    def test_imagenet(self):
        parser = ArgumentParser()
        parser = TinyImageNetDataModule.add_argparse_args(parser)

        args = parser.parse_args("")
        _ = TinyImageNetDataModule(**vars(args))
