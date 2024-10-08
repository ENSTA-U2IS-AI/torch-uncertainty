from pathlib import Path
from typing import Literal

from torch_uncertainty.models import deep_ensembles
from torch_uncertainty.routines.classification import ClassificationRoutine
from torch_uncertainty.utils import get_version

from . import ResNetBaseline, VGGBaseline, WideResNetBaseline


class DeepEnsemblesBaseline(ClassificationRoutine):
    backbones = {
        "resnet": ResNetBaseline,
        "vgg": VGGBaseline,
        "wideresnet": WideResNetBaseline,
    }

    def __init__(
        self,
        num_classes: int,
        log_path: str | Path,
        checkpoint_ids: list[int],
        backbone: Literal["resnet", "vgg", "wideresnet"],
        eval_ood: bool = False,
        eval_shift: bool = False,
        eval_grouping_loss: bool = False,
        ood_criterion: Literal[
            "msp", "logit", "energy", "entropy", "mi", "vr"
        ] = "msp",
        log_plots: bool = False,
        calibration_set: Literal["val", "test"] = "val",
    ) -> None:
        log_path = Path(log_path)

        backbone_cls = self.backbones[backbone]

        models = []
        for version in checkpoint_ids:  # coverage: ignore
            ckpt_file, hparams_file = get_version(
                root=log_path, version=version
            )
            trained_model = backbone_cls.load_from_checkpoint(
                checkpoint_path=ckpt_file,
                hparams_file=hparams_file,
                loss=None,
                optim_recipe=None,
            ).eval()
            models.append(trained_model.model)
        de = deep_ensembles(models=models)

        super().__init__(  # coverage: ignore
            num_classes=num_classes,
            model=de,
            loss=None,
            is_ensemble=de.num_estimators > 1,
            eval_ood=eval_ood,
            eval_shift=eval_shift,
            eval_grouping_loss=eval_grouping_loss,
            ood_criterion=ood_criterion,
            log_plots=log_plots,
            calibration_set=calibration_set,
        )
        self.save_hyperparameters()  # coverage: ignore
