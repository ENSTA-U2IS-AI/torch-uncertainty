import copy
from typing import Literal

import torch
from torch import nn
from torch.distributions import Distribution

from torch_uncertainty.utils.distributions import cat_dist


class _DeepEnsembles(nn.Module):
    def __init__(
        self,
        models: list[nn.Module],
    ) -> None:
        """Create a classification deep ensembles from a list of models."""
        super().__init__()
        self.core_models = nn.ModuleList(models)
        self.num_estimators = len(models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Return the logits of the ensemble.

        Args:
            x (Tensor): The input of the model.

        Returns:
            Tensor: The output of the model with shape :math:`(N \times B, C)`,
                where :math:`B` is the batch size, :math:`N` is the number of
                estimators, and :math:`C` is the number of classes.
        """
        return torch.cat(
            [model.forward(x) for model in self.core_models], dim=0
        )


class _RegDeepEnsembles(_DeepEnsembles):
    def __init__(
        self,
        probabilistic: bool,
        models: list[nn.Module],
    ) -> None:
        """Create a regression deep ensembles from a list of models."""
        super().__init__(models)
        self.probabilistic = probabilistic

    def forward(self, x: torch.Tensor) -> Distribution:
        r"""Return the logits of the ensemble.

        Args:
            x (Tensor): The input of the model.

        Returns:
            Distribution:
        """
        if self.probabilistic:
            return cat_dist(
                [model.forward(x) for model in self.core_models], dim=0
            )
        return super().forward(x)


def deep_ensembles(
    models: list[nn.Module] | nn.Module,
    num_estimators: int | None = None,
    task: Literal[
        "classification", "regression", "segmentation", "pixel_regression"
    ] = "classification",
    probabilistic: bool | None = None,
    reset_model_parameters: bool = False,
) -> _DeepEnsembles:
    """Build a Deep Ensembles out of the original models.

    Args:
        models (list[nn.Module] | nn.Module): The model to be ensembled.
        num_estimators (int | None): The number of estimators in the ensemble.
        task (Literal["classification", "regression", "segmentation", "pixel_regression"]): The model task.
            Defaults to "classification".
        probabilistic (bool): Whether the regression model is probabilistic.
        reset_model_parameters (bool): Whether to reset the model parameters
            when :attr:models is a module or a list of length 1.

    Returns:
        _DeepEnsembles: The ensembled model.

    Raises:
        ValueError: If :attr:num_estimators is not specified and :attr:models
            is a module (or singleton list).
        ValueError: If :attr:num_estimators is less than 2 and :attr:models is
            a module (or singleton list).
        ValueError: If :attr:num_estimators is defined while :attr:models is
            a (non-singleton) list.

    References:
        Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell.
        Simple and scalable predictive uncertainty estimation using deep
        ensembles. In NeurIPS, 2017.
    """
    if isinstance(models, list) and len(models) == 0:
        raise ValueError("Models must not be an empty list.")
    if (isinstance(models, list) and len(models) == 1) or isinstance(
        models, nn.Module
    ):
        if num_estimators is None:
            raise ValueError(
                "if models is a module, num_estimators must be specified."
            )
        if num_estimators < 2:
            raise ValueError(
                f"num_estimators must be at least 2. Got {num_estimators}."
            )

        if isinstance(models, list):
            models = models[0]

        models = [copy.deepcopy(models) for _ in range(num_estimators)]

        if reset_model_parameters:
            for model in models:
                for layer in model.children():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()

    elif (
        isinstance(models, list)
        and len(models) > 1
        and num_estimators is not None
    ):
        raise ValueError(
            "num_estimators must be None if you provided a non-singleton list."
        )

    if task in ("classification", "segmentation"):
        return _DeepEnsembles(models=models)
    if task in ("regression", "pixel_regression"):
        if probabilistic is None:
            raise ValueError(
                "probabilistic must be specified for regression models."
            )
        return _RegDeepEnsembles(probabilistic=probabilistic, models=models)
    raise ValueError(f"Unknown task: {task}.")
