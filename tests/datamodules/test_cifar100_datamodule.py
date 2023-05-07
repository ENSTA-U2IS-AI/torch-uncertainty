# fmt:off
from argparse import ArgumentParser

from torch_uncertainty.datamodules import CIFAR100DataModule


# fmt:on
class TestCIFAR100DataModule:
    """Testing the CIFAR100DataModule datamodule class."""

    def test_cifar100_cutout(self):
        parser = ArgumentParser()
        parser = CIFAR100DataModule.add_argparse_args(parser)

        # Simulate that cutout is set to 8
        args = parser.parse_args("")
        args.cutout = 8

        _ = CIFAR100DataModule(**vars(args))
