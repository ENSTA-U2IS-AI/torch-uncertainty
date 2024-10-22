import copy

from torch import nn

from torch_uncertainty.layers.distributions import (
    LaplaceLayer,
    NormalInverseGammaLayer,
    NormalLayer,
)
from torch_uncertainty.models import EMA, SWA, deep_ensembles
from torch_uncertainty.optim_recipes import optim_cifar10_resnet18
from torch_uncertainty.post_processing import TemperatureScaler
from torch_uncertainty.routines import (
    ClassificationRoutine,
    PixelRegressionRoutine,
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
        loss: nn.Module,
        baseline_type: str = "single",
        optim_recipe=optim_cifar10_resnet18,
        with_feats: bool = True,
        ood_criterion: str = "msp",
        eval_ood: bool = False,
        eval_shift: bool = False,
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
        no_mixup_params: bool = False,
        ema: bool = False,
        swa: bool = False,
    ) -> ClassificationRoutine:
        model = dummy_model(
            in_channels=in_channels,
            num_classes=num_classes,
            with_feats=with_feats,
        )
        if ema:
            model = EMA(model, momentum=0.99)
        if swa:
            model = SWA(model, cycle_start=0, cycle_length=1)
        if not no_mixup_params:
            mixup_params = {
                "mixup_alpha": mixup_alpha,
                "cutmix_alpha": cutmix_alpha,
                "mixtype": mixtype,
                "mixmode": mixmode,
                "dist_sim": dist_sim,
                "kernel_tau_max": kernel_tau_max,
                "kernel_tau_std": kernel_tau_std,
            }
        else:
            mixup_params = None
        if baseline_type == "single":
            return ClassificationRoutine(
                num_classes=num_classes,
                model=model,
                loss=loss,
                format_batch_fn=nn.Identity(),
                log_plots=True,
                optim_recipe=optim_recipe(model),
                is_ensemble=False,
                mixup_params=mixup_params,
                ood_criterion=ood_criterion,
                eval_ood=eval_ood,
                eval_shift=eval_shift,
                eval_grouping_loss=eval_grouping_loss,
                post_processing=TemperatureScaler() if calibrate else None,
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
            is_ensemble=True,
            ood_criterion=ood_criterion,
            eval_ood=eval_ood,
            eval_shift=eval_shift,
            eval_grouping_loss=eval_grouping_loss,
            post_processing=TemperatureScaler() if calibrate else None,
            save_in_csv=save_in_csv,
        )


class DummyRegressionBaseline:
    def __new__(
        cls,
        probabilistic: bool,
        in_features: int,
        output_dim: int,
        loss: nn.Module,
        baseline_type: str = "single",
        optim_recipe=None,
        dist_type: str = "normal",
        ema: bool = False,
        swa: bool = False,
    ) -> RegressionRoutine:
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
        if ema:
            model = EMA(model, momentum=0.99)
        if swa:
            model = SWA(model, cycle_start=0, cycle_length=1)

        if baseline_type == "single":
            return RegressionRoutine(
                probabilistic=probabilistic,
                output_dim=output_dim,
                model=model,
                loss=loss,
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
            is_ensemble=True,
            optim_recipe=optim_recipe(model),
            format_batch_fn=RepeatTarget(2),
        )


class DummySegmentationBaseline:
    def __new__(
        cls,
        in_channels: int,
        num_classes: int,
        image_size: int,
        loss: nn.Module,
        baseline_type: str = "single",
        optim_recipe=None,
        metric_subsampling_rate: float = 1,
        log_plots: bool = False,
        ema: bool = False,
        swa: bool = False,
    ) -> SegmentationRoutine:
        model = dummy_segmentation_model(
            in_channels=in_channels,
            num_classes=num_classes,
            image_size=image_size,
        )
        if ema:
            model = EMA(model, momentum=0.99)
        if swa:
            model = SWA(model, cycle_start=0, cycle_length=2)

        if baseline_type == "single":
            return SegmentationRoutine(
                num_classes=num_classes,
                model=model,
                loss=loss,
                format_batch_fn=None,
                optim_recipe=optim_recipe(model),
                metric_subsampling_rate=metric_subsampling_rate,
                log_plots=log_plots,
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
            optim_recipe=optim_recipe(model),
            metric_subsampling_rate=metric_subsampling_rate,
            log_plots=log_plots,
        )


class DummyPixelRegressionBaseline:
    def __new__(
        cls,
        probabilistic: bool,
        in_channels: int,
        output_dim: int,
        image_size: int,
        loss: nn.Module,
        dist_type: str = "normal",
        baseline_type: str = "single",
        optim_recipe=None,
        ema: bool = False,
        swa: bool = False,
    ) -> PixelRegressionRoutine:
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

        model = dummy_segmentation_model(
            num_classes=num_classes,
            in_channels=in_channels,
            image_size=image_size,
            last_layer=last_layer,
        )
        if ema:
            model = EMA(model, momentum=0.99)
        if swa:
            model = SWA(model, cycle_start=0, cycle_length=1)

        if baseline_type == "single":
            return PixelRegressionRoutine(
                probabilistic=probabilistic,
                output_dim=output_dim,
                model=model,
                loss=loss,
                optim_recipe=optim_recipe(model),
            )

        # baseline_type == "ensemble":
        model = deep_ensembles(
            [model, copy.deepcopy(model)],
            task="pixel_regression",
            probabilistic=probabilistic,
        )
        return PixelRegressionRoutine(
            probabilistic=probabilistic,
            output_dim=output_dim,
            model=model,
            loss=loss,
            format_batch_fn=RepeatTarget(2),
            is_ensemble=True,
            optim_recipe=optim_recipe(model),
        )
