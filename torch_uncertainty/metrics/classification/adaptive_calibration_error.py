from typing import Any, Literal

import numpy as np
import torch
from torch import Tensor
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel


def _hist_edges_equal(x: Tensor, num_bins: int):
    npt = len(x)
    return np.interp(
        np.linspace(0, npt, num_bins + 1), np.arange(npt), np.sort(x)
    )


def _ace_compute(
    confidences: Tensor,
    accuracies: Tensor,
    num_bins: int,
    norm: Literal["l1", "l2", "max"],
) -> Tensor:
    """Compute metric."""
    # Get edges
    bin_boundaries = np.histogram(
        a=confidences.cpu().detach(),
        bins=_hist_edges_equal(confidences.cpu().detach(), num_bins),
    )[1]

    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    adaptive_ece = torch.zeros(1, device=confidences.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
        in_bin = (confidences > bin_lower.item()) * (
            confidences < bin_upper.item()
        )
        prop_bin = in_bin.float().mean()
        if prop_bin.item() > 0:
            acc_bin = accuracies[in_bin].float().mean()
            conf_bin = confidences[in_bin].mean()
            if norm == "l1":
                ace = torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
                adaptive_ece += ace
            if norm == "max":
                ace = torch.max(torch.abs(acc_bin - conf_bin))
                adaptive_ece = torch.max(adaptive_ece, ace)
            if norm == "l2":
                ace = torch.sum(torch.pow(acc_bin - conf_bin, 2) * prop_bin)
                ace = torch.sqrt(ace) if ace > 0 else torch.tensor(0)
                adaptive_ece += ace
    return adaptive_ece


class BinaryAdaptiveCalibrationError(Metric):
    r"""`Adaptive Top-label Calibration Error` for binary tasks."""

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
        self.n_bins = n_bins
        self.norm = norm
        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")

    def update(self, probs: Tensor, targets: Tensor) -> None:
        """Update metric states with predictions and targets."""
        confidences, preds = torch.max(probs, 1 - probs), torch.round(probs)
        accuracies = preds == targets
        self.confidences.append(confidences)
        self.accuracies.append(accuracies)

    def compute(self) -> Tensor:
        """Compute metric."""
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)
        return _ace_compute(
            confidences, accuracies, self.n_bins, norm=self.norm
        )


class MulticlassAdaptiveCalibrationError(Metric):
    r"""`Adaptive Top-label Calibration Error` for multiclass tasks."""

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
        self.n_bins = n_bins
        self.norm = norm
        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")

    def update(self, probs: Tensor, targets: Tensor) -> None:
        """Update metric states with predictions and targets."""
        confidences, preds = torch.max(probs, 1)
        accuracies = preds == targets
        self.confidences.append(confidences)
        self.accuracies.append(accuracies)

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
        if task == ClassificationTaskNoMultilabel.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
                )
            return MulticlassAdaptiveCalibrationError(num_classes, **kwargs)
        raise ValueError(f"Not handled value: {task}")
