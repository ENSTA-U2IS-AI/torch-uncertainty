# fmt:off
from argparse import ArgumentParser

from torch_uncertainty.datamodules import MNISTDataModule


# fmt:on
class TestMNISTDataModule:
    """Testing the MNISTDataModule datamodule class."""

    def test_mnist_cutout(self):
        parser = ArgumentParser()
        parser = MNISTDataModule.add_argparse_args(parser)

        # Simulate that cutout is set to 16
        args = parser.parse_args("")
        args.cutout = 16

        _ = MNISTDataModule(**vars(args))
