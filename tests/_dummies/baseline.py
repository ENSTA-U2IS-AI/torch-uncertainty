# fmt: off
from argparse import ArgumentParser
from typing import Any

import torch.nn as nn
from pytorch_lightning import LightningModule

from torch_uncertainty.routines.classification import (
    ClassificationEnsemble,
    ClassificationSingle,
)
from torch_uncertainty.routines.regression import (
    RegressionEnsemble,
    RegressionSingle,
)

from .model import dummy_model


# fmt: on
class DummyClassificationBaseline:
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
            kwargs["num_estimators"] = 2
            return ClassificationEnsemble(
                num_classes=num_classes,
                model=model,
                loss=loss,
                optimization_procedure=optimization_procedure,
                **kwargs,
            )

    @classmethod
    def add_model_specific_args(
        cls,
        parser: ArgumentParser,
    ) -> ArgumentParser:
        parser = ClassificationEnsemble.add_model_specific_args(parser)
        return parser


class DummyRegressionBaseline:
    def __new__(
        cls,
        in_features: int,
        out_features: int,
        loss: nn.Module,
        optimization_procedure: Any,
        baseline_type: str = "single",
        dist_estimation: bool = False,
        **kwargs,
    ) -> LightningModule:
        model = dummy_model(
            in_channels=in_features,
            num_classes=out_features,
            num_estimators=1 + int(baseline_type == "ensemble"),
        )

        if baseline_type == "single":
            return RegressionSingle(
                out_features=out_features,
                model=model,
                loss=loss,
                optimization_procedure=optimization_procedure,
                dist_estimation=dist_estimation,
                **kwargs,
            )
        elif baseline_type == "ensemble":
            kwargs["num_estimators"] = 2
            return RegressionEnsemble(
                model=model,
                loss=loss,
                optimization_procedure=optimization_procedure,
                dist_estimation=dist_estimation,
                mode="mean",
                out_features=out_features,
                **kwargs,
            )

    @classmethod
    def add_model_specific_args(
        cls,
        parser: ArgumentParser,
    ) -> ArgumentParser:
        parser = ClassificationEnsemble.add_model_specific_args(parser)
        return parser
