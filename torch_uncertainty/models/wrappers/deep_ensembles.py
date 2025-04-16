import copy
import warnings
from typing import Literal

import torch
from torch import nn


class _DeepEnsembles(nn.Module):
    def __init__(
        self,
        models: list[nn.Module],
        store_on_cpu: bool = False,
    ) -> None:
        """Create a classification deep ensembles from a list of models."""
        super().__init__()
        self.core_models = nn.ModuleList(models)
        self.num_estimators = len(models)
        self.store_on_cpu = store_on_cpu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Return the logits of the ensemble.

        Args:
            x (Tensor): The input of the model.

        Returns:
            Tensor: The output of the model with shape :math:`(N \times B, C)`,
                where :math:`B` is the batch size, :math:`N` is the number of
                estimators, and :math:`C` is the number of classes.
        """
        if self.store_on_cpu:
            preds = torch.tensor([], device=x.device)
            for model in self.core_models:
                model.to(x.device)
                preds = torch.cat([preds, model.forward(x)], dim=0)
                model.to("cpu")
            return preds
        return torch.cat([model.forward(x) for model in self.core_models], dim=0)

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(*args, **kwargs)

        if self.store_on_cpu:
            device = torch.device("cpu")

        if dtype is not None:
            if not (dtype.is_floating_point or dtype.is_complex):
                raise TypeError(
                    "nn.Module.to only accepts floating point or complex "
                    f"dtypes, but got desired dtype={dtype}"
                )
            if dtype.is_complex:
                warnings.warn(
                    "Complex modules are a new feature under active development whose design may change, "
                    "and some modules might not work as expected when using complex tensors as parameters or buffers. "
                    "Please file an issue at https://github.com/pytorch/pytorch/issues/new?template=bug-report.yml "
                    "if a complex module does not work as expected.",
                    stacklevel=2,
                )

        def convert(t):
            try:
                if convert_to_format is not None and t.dim() in (4, 5):
                    return t.to(
                        device,
                        dtype if t.is_floating_point() or t.is_complex() else None,
                        non_blocking,
                        memory_format=convert_to_format,
                    )
                return t.to(
                    device,
                    dtype if t.is_floating_point() or t.is_complex() else None,
                    non_blocking,
                )
            except NotImplementedError as e:
                if str(e) == "Cannot copy out of meta tensor; no data!":
                    raise NotImplementedError(
                        f"{e} Please use torch.nn.Module.to_empty() instead of torch.nn.Module.to() "
                        f"when moving module from meta to a different device."
                    ) from None
                raise

        return self._apply(convert)


class _RegDeepEnsembles(_DeepEnsembles):
    def __init__(
        self,
        probabilistic: bool,
        models: list[nn.Module],
        store_on_cpu: bool = False,
    ) -> None:
        """Create a regression deep ensembles from a list of models."""
        super().__init__(models=models, store_on_cpu=store_on_cpu)
        self.probabilistic = probabilistic

    def forward(self, x: torch.Tensor) -> torch.Tensor | dict[str, torch.Tensor]:
        r"""Return the logits of the ensemble.

        Args:
            x (Tensor): The input of the model.

        Returns:
            Tensor | dict[str, Tensor]: The output of the model with shape :math:`(N \times B, *)`
                where :math:`B` is the batch size, :math:`N` is the number of estimators, and
                :math:`*` is any other dimension.
        """
        if self.probabilistic:
            if self.store_on_cpu:
                out = []
                for model in self.core_models:
                    model.to(x.device)
                    out.append(model.forward(x))
                    model.to("cpu")
            else:
                out = [model.forward(x) for model in self.core_models]
            key_set = {tuple(o.keys()) for o in out}
            if len(key_set) != 1:
                raise ValueError("The output of the models must have the same keys.")
            return {k: torch.cat([o[k] for o in out], dim=0) for k in key_set.pop()}
        return super().forward(x)


def deep_ensembles(
    models: list[nn.Module] | nn.Module,
    num_estimators: int | None = None,
    task: Literal[
        "classification", "regression", "segmentation", "pixel_regression"
    ] = "classification",
    probabilistic: bool | None = None,
    reset_model_parameters: bool = True,
    store_on_cpu: bool = False,
) -> _DeepEnsembles:
    """Build a Deep Ensembles out of the original models.

    Args:
        models (list[nn.Module] | nn.Module): The model to be ensembled.
        num_estimators (int | None): The number of estimators in the ensemble.
        task (Literal["classification", "regression", "segmentation", "pixel_regression"]): The model task.
            Defaults to "classification".
        probabilistic (bool): Whether the regression model is probabilistic.
        reset_model_parameters (bool): Whether to reset the model parameters
            when :attr:models is a module or a list of length 1. Defaults to ``True``.
        store_on_cpu (bool): Whether to store the models on CPU. Defaults to ``False``.
            This is useful for large models that do not fit in GPU memory. Only one
            model will be stored on GPU at a time during forward. The rest will be stored on CPU.

    Returns:
        _DeepEnsembles: The ensembled model.

    Raises:
        ValueError: If :attr:num_estimators is not specified and :attr:models
            is a module (or singleton list).
        ValueError: If :attr:num_estimators is less than 2 and :attr:models is
            a module (or singleton list).
        ValueError: If :attr:num_estimators is defined while :attr:models is
            a (non-singleton) list.

    Warning:
        The :attr:`store_on_cpu` option is not supported for training. It is
        only supported for inference.

    References:
        Balaji Lakshminarayanan, Alexander Pritzel, and Charles Blundell.
        Simple and scalable predictive uncertainty estimation using deep
        ensembles. In NeurIPS, 2017.
    """
    if isinstance(models, list) and len(models) == 0:
        raise ValueError("Models must not be an empty list.")
    if (isinstance(models, list) and len(models) == 1) or isinstance(models, nn.Module):
        if num_estimators is None:
            raise ValueError("if models is a module, num_estimators must be specified.")
        if num_estimators < 2:
            raise ValueError(f"num_estimators must be at least 2. Got {num_estimators}.")

        if isinstance(models, list):
            models = models[0]

        models = [copy.deepcopy(models) for _ in range(num_estimators)]

        if reset_model_parameters:
            for model in models:
                for layer in model.modules():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()

    elif isinstance(models, list) and len(models) > 1 and num_estimators is not None:
        raise ValueError("num_estimators must be None if you provided a non-singleton list.")

    if task in ("classification", "segmentation"):
        return _DeepEnsembles(models=models, store_on_cpu=store_on_cpu)
    if task in ("regression", "pixel_regression"):
        if probabilistic is None:
            raise ValueError("probabilistic must be specified for regression models.")
        return _RegDeepEnsembles(
            probabilistic=probabilistic, models=models, store_on_cpu=store_on_cpu
        )
    raise ValueError(f"Unknown task: {task}.")
