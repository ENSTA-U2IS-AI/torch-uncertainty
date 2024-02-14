
from pytorch_lightning import LightningModule
from torch import nn

from torch_uncertainty.routines import ClassificationRoutine, RegressionRoutine
from torch_uncertainty.transforms import RepeatTarget

from .model import dummy_model


class DummyClassificationBaseline:
    def __new__(
        cls,
        num_classes: int,
        in_channels: int,
        loss: type[nn.Module],
        baseline_type: str = "single",
        with_feats: bool = True,
        with_linear: bool = True,
    ) -> LightningModule:
        model = dummy_model(
            in_channels=in_channels,
            num_classes=num_classes,
            num_estimators=1 + int(baseline_type == "ensemble"),
            with_feats=with_feats,
            with_linear=with_linear,
        )

        if baseline_type == "single":
            return ClassificationRoutine(
                num_classes=num_classes,
                model=model,
                loss=loss,
                format_batch_fn=nn.Identity(),
                log_plots=True,
                num_estimators = 1
            )
        # baseline_type == "ensemble":
        return ClassificationRoutine(
            num_classes=num_classes,
            model=model,
            loss=loss,
            format_batch_fn=RepeatTarget(2),
            log_plots=True,
            num_estimators = 2
        )

    # @classmethod
    # def add_model_specific_args(
    #     cls,
    #     parser: ArgumentParser,
    # ) -> ArgumentParser:
    #     return ClassificationEnsemble.add_model_specific_args(parser)


class DummyRegressionBaseline:
    def __new__(
        cls,
        in_features: int,
        out_features: int,
        loss: type[nn.Module],
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
            return RegressionRoutine(
                out_features=out_features,
                model=model,
                loss=loss,
                dist_estimation=dist_estimation,
                num_estimators=1
            )
        # baseline_type == "ensemble":
        kwargs["num_estimators"] = 2
        return RegressionRoutine(
            model=model,
            loss=loss,
            dist_estimation=dist_estimation,
            mode="mean",
            out_features=out_features,
            num_estimators=2
        )

    # @classmethod
    # def add_model_specific_args(
    #     cls,
    #     parser: ArgumentParser,
    # ) -> ArgumentParser:
    #     return ClassificationEnsemble.add_model_specific_args(parser)
