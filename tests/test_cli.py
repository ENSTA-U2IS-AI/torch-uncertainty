# fmt: off
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch.nn as nn

from torch_uncertainty import cli_main, main
from torch_uncertainty.baselines.standard import ResNet
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.optimization_procedures import optim_cifar10_resnet18

from ._dummies import Dummy, DummyDataModule


# fmt: on
class TestCLI:
    """Testing the CLI function."""

    def test_main_summary(self):
        root = Path(__file__).parent.absolute().parents[0]

        parser = ArgumentParser("torch-uncertainty")
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--test", type=int, default=None)
        parser.add_argument("--summary", dest="summary", action="store_true")
        parser.add_argument(
            "--log_graph", dest="log_graph", action="store_true"
        )
        parser.add_argument(
            "--channels_last",
            action="store_true",
            help="Use channels last memory format",
        )

        datamodule = CIFAR10DataModule
        network = ResNet
        parser = pl.Trainer.add_argparse_args(parser)
        parser = datamodule.add_argparse_args(parser)
        parser = network.add_model_specific_args(parser)

        # Simulate that summary is True & the only argument
        args = parser.parse_args(["--no-imagenet_structure"])
        args.summary = True

        main(
            network,
            datamodule,
            nn.CrossEntropyLoss,
            optim_cifar10_resnet18,
            root,
            "std",
            args,
        )

    def test_cli_main(self):
        root = Path(__file__).parent.absolute().parents[0]
        cli_main(
            Dummy,
            DummyDataModule,
            nn.CrossEntropyLoss,
            optim_cifar10_resnet18,
            root,
            "dummy",
        )
