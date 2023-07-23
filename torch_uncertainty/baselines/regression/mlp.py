# fmt: off
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from pytorch_lightning.core.saving import (
    load_hparams_from_tags_csv,
    load_hparams_from_yaml,
)

from ...models.mlp import mlp, packed_mlp
from ...routines.regression import RegressionEnsemble, RegressionSingle
from ..utils.parser_addons import add_packed_specific_args


# fmt: on
class MLP:
    r"""MLP baseline for regression providing support for various versions."""

    single = ["vanilla"]
    ensemble = ["packed"]
    versions = {"vanilla": mlp, "packed": packed_mlp}

    def __new__(
        cls,
        num_outputs: int,
        in_features: int,
        loss: nn.Module,
        optimization_procedure: Any,
        version: Literal["vanilla", "packed"],
        hidden_dims: List[int],
        dist_estimation: bool,
        num_estimators: Optional[int] = 1,
        alpha: Optional[float] = None,
        gamma: int = 1,
        **kwargs,
    ) -> LightningModule:
        params = {
            "in_features": in_features,
            "num_outputs": num_outputs,
            "hidden_dims": hidden_dims,
        }

        if version == "packed":
            params |= {
                "alpha": alpha,
                "num_estimators": num_estimators,
                "gamma": gamma,
            }

        if version not in cls.versions.keys():
            raise ValueError(f"Unknown version: {version}")

        model = cls.versions[version](**params)

        kwargs.update(params)
        kwargs.update({"version": version})
        # routine specific parameters
        if version in cls.single:
            return RegressionSingle(
                model=model,
                loss=loss,
                optimization_procedure=optimization_procedure,
                dist_estimation=dist_estimation,
                **kwargs,
            )
        elif version in cls.versions.keys():
            return RegressionEnsemble(
                model=model,
                loss=loss,
                optimization_procedure=optimization_procedure,
                dist_estimation=dist_estimation,
                mode="mean",
                **kwargs,
            )

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        hparams_file: Union[str, Path],
        **kwargs,
    ) -> LightningModule:  # coverage: ignore
        if hparams_file is not None:
            extension = str(hparams_file).split(".")[-1]
            if extension.lower() == "csv":
                hparams = load_hparams_from_tags_csv(hparams_file)
            elif extension.lower() in ("yml", "yaml"):
                hparams = load_hparams_from_yaml(hparams_file)
            else:
                raise ValueError(
                    ".csv, .yml or .yaml is required for `hparams_file`"
                )

        hparams.update(kwargs)
        checkpoint = torch.load(checkpoint_path)
        obj = cls(**hparams)
        obj.load_state_dict(checkpoint["state_dict"])
        return obj

    @classmethod
    def add_model_specific_args(cls, parser: ArgumentParser) -> ArgumentParser:
        parser = RegressionEnsemble.add_model_specific_args(parser)
        parser = add_packed_specific_args(parser)
        parser.add_argument(
            "--version",
            type=str,
            choices=cls.versions.keys(),
            default="vanilla",
            help=f"Variation of MLP. Choose among: {cls.versions.keys()}",
        )
        return parser
