import copy

from pytorch_lightning import LightningModule
from torch import nn

from torch_uncertainty.layers.distributions import (
    IndptLaplaceLayer,
    IndptNormalLayer,
)
from torch_uncertainty.models.deep_ensembles import deep_ensembles
from torch_uncertainty.routines import ClassificationRoutine, RegressionRoutine
from torch_uncertainty.transforms import RepeatTarget

from .model import dummy_model


class DummyClassificationBaseline:
    def __new__(
        cls,
        num_classes: int,
        in_channels: int,
        loss: type[nn.Module],
        ensemble=False,
        optim_recipe=None,
        with_feats: bool = True,
        with_linear: bool = True,
        ood_criterion: str = "msp",
        eval_ood: bool = False,
        eval_grouping_loss: bool = False,
    ) -> LightningModule:
        model = dummy_model(
            in_channels=in_channels,
            num_classes=num_classes,
            num_estimators=1 + int(ensemble),
            with_feats=with_feats,
            with_linear=with_linear,
        )

        if not ensemble:
            return ClassificationRoutine(
                num_classes=num_classes,
                model=model,
                loss=loss,
                format_batch_fn=nn.Identity(),
                log_plots=True,
                optim_recipe=optim_recipe,
                num_estimators=1,
                ood_criterion=ood_criterion,
                eval_ood=eval_ood,
                eval_grouping_loss=eval_grouping_loss,
            )
        # ensemble
        return ClassificationRoutine(
            num_classes=num_classes,
            model=model,
            loss=loss,
            optim_recipe=optim_recipe,
            format_batch_fn=RepeatTarget(2),
            log_plots=True,
            num_estimators=2,
            ood_criterion=ood_criterion,
            eval_ood=eval_ood,
            eval_grouping_loss=eval_grouping_loss,
        )


class DummyRegressionBaseline:
    def __new__(
        cls,
        probabilistic: bool,
        in_features: int,
        num_outputs: int,
        loss: type[nn.Module],
        baseline_type: str = "single",
        optim_recipe=None,
        dist_type: str = "normal",
    ) -> LightningModule:
        model = dummy_model(
            in_channels=in_features,
            num_classes=num_outputs * 2 if probabilistic else num_outputs,
            num_estimators=1,
            last_layer=(
                IndptNormalLayer(num_outputs)
                if dist_type == "normal"
                else IndptLaplaceLayer(num_outputs)
            )
            if probabilistic
            else nn.Identity(),
        )
        if baseline_type == "single":
            return RegressionRoutine(
                probabilistic=probabilistic,
                num_outputs=num_outputs,
                model=model,
                loss=loss,
                num_estimators=1,
                optim_recipe=optim_recipe,
            )
        # baseline_type == "ensemble":
        model = deep_ensembles(
            [model, copy.deepcopy(model)],
            task="regression",
            probabilistic=probabilistic,
        )
        return RegressionRoutine(
            probabilistic=probabilistic,
            num_outputs=num_outputs,
            model=model,
            loss=loss,
            num_estimators=2,
            optim_recipe=optim_recipe,
            format_batch_fn=RepeatTarget(2),
        )
