from argparse import ArgumentParser
from typing import Any

from pytorch_lightning import LightningModule
from torch import nn

from torch_uncertainty.routines.classification import (
    ClassificationEnsemble,
    ClassificationSingle,
)
from torch_uncertainty.routines.regression import (
    RegressionEnsemble,
    RegressionSingle,
)
from torch_uncertainty.transforms import RepeatTarget

from .model import dummy_model


class DummyClassificationBaseline:
    def __new__(
        cls,
        num_classes: int,
        in_channels: int,
        loss: type[nn.Module],
        optimization_procedure: Any,
        baseline_type: str = "single",
        with_feats: bool = True,
        with_linear: bool = True,
        **kwargs,
    ) -> LightningModule:
        model = dummy_model(
            in_channels=in_channels,
            num_classes=num_classes,
            num_estimators=1 + int(baseline_type == "ensemble"),
            with_feats=with_feats,
            with_linear=with_linear,
        )

        if baseline_type == "single":
            return ClassificationSingle(
                num_classes=num_classes,
                model=model,
                loss=loss,
                optimization_procedure=optimization_procedure,
                format_batch_fn=nn.Identity(),
                log_plots=True,
                **kwargs,
            )
        # baseline_type == "ensemble":
        kwargs["num_estimators"] = 2
        return ClassificationEnsemble(
            num_classes=num_classes,
            model=model,
            loss=loss,
            optimization_procedure=optimization_procedure,
            format_batch_fn=RepeatTarget(2),
            log_plots=True,
            **kwargs,
        )

    @classmethod
    def add_model_specific_args(
        cls,
        parser: ArgumentParser,
    ) -> ArgumentParser:
        return ClassificationEnsemble.add_model_specific_args(parser)


class DummyRegressionBaseline:
    def __new__(
        cls,
        in_features: int,
        out_features: int,
        loss: type[nn.Module],
        optimization_procedure: Any,
        baseline_type: str = "single",
        dist_estimation: int = 1,
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
        # baseline_type == "ensemble":
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
        return ClassificationEnsemble.add_model_specific_args(parser)
