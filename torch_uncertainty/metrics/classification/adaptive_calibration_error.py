from typing import Any, Literal

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.classification.calibration_error import (
    _binary_calibration_error_arg_validation,
    _multiclass_calibration_error_arg_validation,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel


def _equal_binning_bucketize(
    confidences: Tensor, accuracies: Tensor, num_bins: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute bins for the adaptive calibration error.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top1
            prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        num_bins: Number of bins to use when computing adaptive calibration
            error.

    Returns:
        tuple with binned accuracy, binned confidence and binned probabilities
    """
    confidences, indices = torch.sort(confidences)
    accuracies = accuracies[indices]
    acc_bin, conf_bin = (
        accuracies.tensor_split(num_bins),
        confidences.tensor_split(num_bins),
    )
    count_bin = torch.as_tensor(
        [len(cb) for cb in conf_bin],
        dtype=confidences.dtype,
        device=confidences.device,
    )
    return (
        pad_sequence(acc_bin, batch_first=True).sum(1) / count_bin,
        pad_sequence(conf_bin, batch_first=True).sum(1) / count_bin,
        torch.as_tensor(count_bin) / len(confidences),
    )


def _ace_compute(
    confidences: Tensor,
    accuracies: Tensor,
    num_bins: int,
    norm: Literal["l1", "l2", "max"] = "l1",
    debias: bool = False,
) -> Tensor:
    """Compute the adaptive calibration error given the provided number of bins
        and norm.

    Args:
        confidences: The confidence (i.e. predicted prob) of the top1
            prediction.
        accuracies: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
        num_bins: Number of bins to use when computing adaptive calibration
            error.
        norm: Norm function to use when computing calibration error. Defaults
            to "l1".
        debias: Apply debiasing to L2 norm computation as in
            `Verified Uncertainty Calibration`. Defaults to False.

    Returns:
        Tensor: Adaptive Calibration error scalar.
    """
    with torch.no_grad():
        acc_bin, conf_bin, prop_bin = _equal_binning_bucketize(
            confidences, accuracies, num_bins
        )

    if norm == "l1":
        return torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
    if norm == "max":
        ace = torch.max(torch.abs(acc_bin - conf_bin))
    if norm == "l2":
        ace = torch.sum(torch.pow(acc_bin - conf_bin, 2) * prop_bin)
        if debias:  # coverage: ignore
            debias_bins = (acc_bin * (acc_bin - 1) * prop_bin) / (
                prop_bin * accuracies.size()[0] - 1
            )
            ace += torch.sum(
                torch.nan_to_num(debias_bins)
            )  # replace nans with zeros if nothing appeared in a bin
        return torch.sqrt(ace) if ace > 0 else torch.tensor(0)
    return ace


class BinaryAdaptiveCalibrationError(Metric):
    r"""Adaptive Top-label Calibration Error for binary tasks."""

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    confidences: list[Tensor]
    accuracies: list[Tensor]

    def __init__(
        self,
        n_bins: int = 10,
        norm: Literal["l1", "l2", "max"] = "l1",
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if ignore_index is not None:  # coverage: ignore
            raise ValueError(
                "ignore_index is not supported for multiclass tasks."
            )

        if validate_args:
            _binary_calibration_error_arg_validation(n_bins, norm, ignore_index)
        self.n_bins = n_bins
        self.norm = norm

        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")

    def update(self, probs: Tensor, targets: Tensor) -> None:
        """Update metric states with predictions and targets."""
        confidences, preds = torch.max(probs, 1 - probs), torch.round(probs)
        accuracies = preds == targets
        self.confidences.append(confidences.float())
        self.accuracies.append(accuracies.float())

    def compute(self) -> Tensor:
        """Compute metric."""
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        return _ace_compute(
            confidences, accuracies, self.n_bins, norm=self.norm
        )


class MulticlassAdaptiveCalibrationError(Metric):
    r"""Adaptive Top-label Calibration Error for multiclass tasks."""

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    confidences: list[Tensor]
    accuracies: list[Tensor]

    def __init__(
        self,
        num_classes: int,
        n_bins: int = 10,
        norm: Literal["l1", "l2", "max"] = "l1",
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        if ignore_index is not None:  # coverage: ignore
            raise ValueError(
                "ignore_index is not supported for multiclass tasks."
            )

        if validate_args:
            _multiclass_calibration_error_arg_validation(
                num_classes, n_bins, norm, ignore_index
            )
        self.n_bins = n_bins
        self.norm = norm

        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")

    def update(self, probs: Tensor, targets: Tensor) -> None:
        """Update metric states with predictions and targets."""
        confidences, preds = torch.max(probs, 1)
        accuracies = preds == targets
        self.confidences.append(confidences.float())
        self.accuracies.append(accuracies.float())

    def compute(self) -> Tensor:
        """Compute metric."""
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        return _ace_compute(
            confidences, accuracies, self.n_bins, norm=self.norm
        )


class AdaptiveCalibrationError:
    """`Adaptive Top-label Calibration Error`.

    Reference:
        Nixon et al. Measuring calibration in deep learning. In CVPRW, 2019.
    """

    def __new__(
        cls,
        task: Literal["binary", "multiclass"],
        num_bins: int = 10,
        norm: Literal["l1", "l2", "max"] = "l1",
        num_classes: int | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        task = ClassificationTaskNoMultilabel.from_str(task)
        kwargs.update(
            {
                "n_bins": num_bins,
                "norm": norm,
                "ignore_index": ignore_index,
                "validate_args": validate_args,
            }
        )
        if task == ClassificationTaskNoMultilabel.BINARY:
            return BinaryAdaptiveCalibrationError(**kwargs)
        # task is ClassificationTaskNoMultilabel.MULTICLASS
        if not isinstance(num_classes, int):
            raise TypeError(
                f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
            )
        return MulticlassAdaptiveCalibrationError(num_classes, **kwargs)
