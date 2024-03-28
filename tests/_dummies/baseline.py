import copy

from pytorch_lightning import LightningModule
from torch import nn

from torch_uncertainty.layers.distributions import (
    LaplaceLayer,
    NormalInverseGammaLayer,
    NormalLayer,
)
from torch_uncertainty.models.deep_ensembles import deep_ensembles
from torch_uncertainty.routines import (
    ClassificationRoutine,
    RegressionRoutine,
    SegmentationRoutine,
)
from torch_uncertainty.transforms import RepeatTarget

from .model import dummy_model, dummy_segmentation_model


class DummyClassificationBaseline:
    def __new__(
        cls,
        num_classes: int,
        in_channels: int,
        loss: type[nn.Module],
        baseline_type: str = "single",
        optim_recipe=None,
        with_feats: bool = True,
        with_linear: bool = True,
        ood_criterion: str = "msp",
        eval_ood: bool = False,
        eval_grouping_loss: bool = False,
        calibrate: bool = False,
        save_in_csv: bool = False,
        mixtype: str = "erm",
        mixmode: str = "elem",
        dist_sim: str = "emb",
        kernel_tau_max: float = 1,
        kernel_tau_std: float = 0.5,
        mixup_alpha: float = 0,
        cutmix_alpha: float = 0,
    ) -> LightningModule:
        model = dummy_model(
            in_channels=in_channels,
            num_classes=num_classes,
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
                optim_recipe=optim_recipe(model),
                num_estimators=1,
                mixtype=mixtype,
                mixmode=mixmode,
                dist_sim=dist_sim,
                kernel_tau_max=kernel_tau_max,
                kernel_tau_std=kernel_tau_std,
                mixup_alpha=mixup_alpha,
                cutmix_alpha=cutmix_alpha,
                ood_criterion=ood_criterion,
                eval_ood=eval_ood,
                eval_grouping_loss=eval_grouping_loss,
                calibration_set="val" if calibrate else None,
                save_in_csv=save_in_csv,
            )
        # baseline_type == "ensemble":
        model = deep_ensembles(
            [model, copy.deepcopy(model)],
            task="classification",
        )
        return ClassificationRoutine(
            num_classes=num_classes,
            model=model,
            loss=loss,
            optim_recipe=optim_recipe(model),
            format_batch_fn=RepeatTarget(2),
            log_plots=True,
            num_estimators=2,
            ood_criterion=ood_criterion,
            eval_ood=eval_ood,
            eval_grouping_loss=eval_grouping_loss,
            calibration_set="val" if calibrate else None,
            save_in_csv=save_in_csv,
        )


class DummyRegressionBaseline:
    def __new__(
        cls,
        probabilistic: bool,
        in_features: int,
        output_dim: int,
        loss: type[nn.Module],
        baseline_type: str = "single",
        optim_recipe=None,
        dist_type: str = "normal",
    ) -> LightningModule:
        if probabilistic:
            if dist_type == "normal":
                last_layer = NormalLayer(output_dim)
                num_classes = output_dim * 2
            elif dist_type == "laplace":
                last_layer = LaplaceLayer(output_dim)
                num_classes = output_dim * 2
            else:  # dist_type == "nig"
                last_layer = NormalInverseGammaLayer(output_dim)
                num_classes = output_dim * 4
        else:
            last_layer = nn.Identity()
            num_classes = output_dim

        model = dummy_model(
            in_channels=in_features,
            num_classes=num_classes,
            last_layer=last_layer,
        )
        if baseline_type == "single":
            return RegressionRoutine(
                probabilistic=probabilistic,
                output_dim=output_dim,
                model=model,
                loss=loss,
                num_estimators=1,
                optim_recipe=optim_recipe(model),
            )
        # baseline_type == "ensemble":
        model = deep_ensembles(
            [model, copy.deepcopy(model)],
            task="regression",
            probabilistic=probabilistic,
        )
        return RegressionRoutine(
            probabilistic=probabilistic,
            output_dim=output_dim,
            model=model,
            loss=loss,
            num_estimators=2,
            optim_recipe=optim_recipe(model),
            format_batch_fn=RepeatTarget(2),
        )


class DummySegmentationBaseline:
    def __new__(
        cls,
        in_channels: int,
        num_classes: int,
        image_size: int,
        loss: type[nn.Module],
        baseline_type: str = "single",
        optim_recipe=None,
    ) -> LightningModule:
        model = dummy_segmentation_model(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
        )

        if baseline_type == "single":
            return SegmentationRoutine(
                num_classes=num_classes,
                model=model,
                loss=loss,
                format_batch_fn=None,
                num_estimators=1,
                optim_recipe=optim_recipe(model),
            )

        # baseline_type == "ensemble":
        model = deep_ensembles(
            [model, copy.deepcopy(model)],
            task="segmentation",
        )
        return SegmentationRoutine(
            num_classes=num_classes,
            model=model,
            loss=loss,
            format_batch_fn=RepeatTarget(2),
            num_estimators=2,
            optim_recipe=optim_recipe(model),
        )
