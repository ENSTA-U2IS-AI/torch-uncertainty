# fmt: off
from argparse import ArgumentParser
from typing import Any

import torch.nn as nn
from pytorch_lightning import LightningModule

from torch_uncertainty.routines.classification import (
    ClassificationEnsemble,
    ClassificationSingle,
)

from .model import dummy_model


# fmt: on
class DummyBaseline:
    r"""LightningModule for Vanilla ResNet.

    Args:
        num_classes (int): Number of classes to predict.
        in_channels (int): Number of input channels.
        loss (torch.nn.Module): Training loss.
        optimization_procedure (Any): Optimization procedure, corresponds to
            what expect the `LightningModule.configure_optimizers()
            <https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#configure-optimizers>`_
            method.
    """

    def __new__(
        cls,
        num_classes: int,
        in_channels: int,
        loss: nn.Module,
        optimization_procedure: Any,
        baseline_type: str = "single",
        **kwargs,
    ) -> LightningModule:
        model = dummy_model(
            in_channels=in_channels,
            num_classes=num_classes,
            num_estimators=1 + int(baseline_type == "ensemble"),
        )

        if baseline_type == "single":
            return ClassificationSingle(
                num_classes=num_classes,
                model=model,
                loss=loss,
                optimization_procedure=optimization_procedure,
                **kwargs,
            )
        elif baseline_type == "ensemble":
            return ClassificationEnsemble(
                num_classes=num_classes,
                num_estimators=2,
                model=model,
                loss=loss,
                optimization_procedure=optimization_procedure,
                **kwargs,
            )

    @staticmethod
    def add_model_specific_args(
        parent_parser: ArgumentParser,
    ) -> ArgumentParser:
        return parent_parser
