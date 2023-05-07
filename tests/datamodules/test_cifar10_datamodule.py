# fmt:off
from argparse import ArgumentParser

from torch_uncertainty.datamodules import CIFAR10DataModule


# fmt:on
class TestCIFAR10DataModule:
    """Testing the CIFAR10DataModule datamodule class."""

    def test_cifar10_cutout(self):
        parser = ArgumentParser()
        parser = CIFAR10DataModule.add_argparse_args(parser)

        # Simulate that cutout is set to 8
        args = parser.parse_args("")
        args.cutout = 16

        _ = CIFAR10DataModule(**vars(args))
