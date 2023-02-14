# fmt: off
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch.nn as nn

from torch_uncertainty import main
from torch_uncertainty.baselines.standard import ResNet
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.optimization_procedures import optim_cifar10_resnet18

# fmt: on


class TestCLI:
    """Testing the VariationRatio metric class."""

    def test_main_summary(self):
        root = Path(__file__).parent.absolute().parents[0]

        parser = ArgumentParser("torch-uncertainty")
        parser.add_argument("--seed", type=int, default=None)
        parser.add_argument("--test", type=int, default=None)
        parser.add_argument("--summary", dest="summary", action="store_true")
        parser.add_argument(
            "--log_graph", dest="log_graph", action="store_true"
        )

        datamodule = CIFAR10DataModule
        network = ResNet
        parser = pl.Trainer.add_argparse_args(parser)
        parser = datamodule.add_argparse_args(parser)
        parser = network.add_model_specific_args(parser)

        # Simulate that summary is True & the only argument
        args = parser.parse_args("")
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
