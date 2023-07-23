# fmt: off
from argparse import ArgumentParser
from pathlib import Path
from typing import List, Literal, Optional, Union

from pytorch_lightning import LightningModule

from ..models import deep_ensembles
from ..routines.classification import ClassificationEnsemble
from ..routines.regression import RegressionEnsemble
from ..utils import get_version
from .classification import VGG, ResNet, WideResNet
from .regression import MLP


# fmt: on
class DeepEnsembles:
    backbones = {
        "mlp": MLP,
        "resnet": ResNet,
        "vgg": VGG,
        "wideresnet": WideResNet,
    }

    def __new__(
        cls,
        task: Literal["classification", "regression"],
        log_path: Union[str, Path],
        versions: List[int],
        backbone: Literal["mlp", "resnet", "vgg", "wideresnet"],
        # num_estimators: int,
        in_channels: Optional[int] = None,
        num_classes: Optional[int] = None,
        use_entropy: bool = False,
        use_logits: bool = False,
        use_mi: bool = False,
        use_variation_ratio: bool = False,
        **kwargs,
    ) -> LightningModule:
        if isinstance(log_path, str):
            log_path = Path(log_path)

        backbone_cls = cls.backbones[backbone]

        models = []
        for version in versions:
            ckpt_file, hparams_file = get_version(
                root=log_path, version=version
            )
            trained_model = backbone_cls.load_from_checkpoint(
                checkpoint_path=ckpt_file,
                hparams_file=hparams_file,
                loss=None,
                optimization_procedure=None,
            ).eval()
            models.append(trained_model.model)

        de = deep_ensembles(models=models)

        if task == "classification":
            return ClassificationEnsemble(
                in_channels=in_channels,
                num_classes=num_classes,
                model=de,
                loss=None,
                optimization_procedure=None,
                num_estimators=de.num_estimators,
                use_entropy=use_entropy,
                use_logits=use_logits,
                use_mi=use_mi,
                use_variation_ratio=use_variation_ratio,
            )
        elif task == "regression":
            return RegressionEnsemble(
                model=de,
                loss=None,
                optimization_procedure=None,
                dist_estimation=True,
                num_estimators=de.num_estimators,
                mode="mean",
            )

    @classmethod
    def add_model_specific_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser = ClassificationEnsemble.add_model_specific_args(parser)
        parser.add_argument(
            "--task",
            type=str,
            choices=["classification", "regression"],
            help="Task to be performed",
        )
        parser.add_argument(
            "--backbone",
            type=str,
            choices=cls.backbones.keys(),
            help="Backbone architecture",
            required=True,
        )
        parser.add_argument(
            "--versions",
            type=int,
            nargs="+",
            help="Versions of the model to be ensembled",
        )
        parser.add_argument(
            "--log_path",
            type=str,
            help="Root directory of the models",
            required=True,
        )
        return parser
