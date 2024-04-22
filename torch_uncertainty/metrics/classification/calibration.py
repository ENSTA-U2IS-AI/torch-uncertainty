from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchmetrics.classification.calibration_error import (
    BinaryCalibrationError,
    MulticlassCalibrationError,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import ClassificationTaskNoMultilabel
from torchmetrics.utilities.plot import _AX_TYPE, _PLOT_OUT_TYPE


class BinaryCE(BinaryCalibrationError):  # noqa: N818
    def plot(self, ax: _AX_TYPE | None = None) -> _PLOT_OUT_TYPE:
        fig, ax = plt.subplots() if ax is None else (None, ax)

        conf = dim_zero_cat(self.confidences)
        acc = dim_zero_cat(self.accuracies)

        bin_width = 1 / self.n_bins

        bin_ids = torch.round(
            torch.clamp(conf * self.n_bins, 1e-5, self.n_bins - 1 - 1e-5)
        )
        val, inverse, counts = bin_ids.unique(
            return_inverse=True, return_counts=True
        )
        val_oh = torch.nn.functional.one_hot(
            val.long(), num_classes=self.n_bins
        )

        # add 1e-6 to avoid division NaNs
        values = (
            val_oh.T.float()
            @ torch.sum(
                acc.unsqueeze(1) * torch.nn.functional.one_hot(inverse).float(),
                0,
            )
            / (val_oh.T @ counts + 1e-6).float()
        )
        counts_all = (val_oh.T @ counts).float()
        total = torch.sum(counts)

        plt.rc("axes", axisbelow=True)
        ax.hist(
            x=[bin_width * i * 100 for i in range(self.n_bins)],
            weights=values * 100,
            bins=[bin_width * i * 100 for i in range(self.n_bins + 1)],
            alpha=0.7,
            linewidth=1,
            edgecolor="#0d559f",
            color="#1f77b4",
        )
        for i, count in enumerate(counts_all):
            ax.text(
                3.0 + 9.9 * i,
                1,
                f"{int(count/total*100)}%",
                fontsize=8,
            )

        ax.plot([0, 100], [0, 100], "--", color="#0d559f")
        plt.grid(True, linestyle="--", alpha=0.7, zorder=0)
        ax.set_xlabel("Top-class Confidence (%)", fontsize=16)
        ax.set_ylabel("Success Rate (%)", fontsize=16)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal", "box")
        fig.tight_layout()
        return fig, ax


class MulticlassCE(MulticlassCalibrationError):  # noqa: N818
    def plot(self, ax: _AX_TYPE | None = None) -> _PLOT_OUT_TYPE:
        fig, ax = plt.subplots() if ax is None else (None, ax)

        conf = dim_zero_cat(self.confidences).cpu()
        acc = dim_zero_cat(self.accuracies).cpu()

        bin_width = 1 / self.n_bins

        bin_ids = torch.round(
            torch.clamp(conf * self.n_bins, 1e-5, self.n_bins - 1 - 1e-5)
        )
        val, inverse, counts = bin_ids.unique(
            return_inverse=True, return_counts=True
        )
        val_oh = torch.nn.functional.one_hot(
            val.long(), num_classes=self.n_bins
        )

        # add 1e-6 to avoid division NaNs
        values = (
            val_oh.T.float()
            @ torch.sum(
                acc.unsqueeze(1) * torch.nn.functional.one_hot(inverse).float(),
                0,
            )
            / (val_oh.T.float() @ counts.float() + 1e-6)
        )
        counts_all = val_oh.T.float() @ counts.float()
        total = torch.sum(counts)

        plt.rc("axes", axisbelow=True)
        ax.hist(
            x=[bin_width * i * 100 for i in range(self.n_bins)],
            weights=values * 100,
            bins=[bin_width * i * 100 for i in range(self.n_bins + 1)],
            alpha=0.7,
            linewidth=1,
            edgecolor="#0d559f",
            color="#1f77b4",
        )
        for i, count in enumerate(counts_all):
            ax.text(
                3.0 + 9.9 * i,
                1,
                f"{int(count/total*100)}%",
                fontsize=8,
            )

        ax.plot([0, 100], [0, 100], "--", color="#0d559f")
        plt.grid(True, linestyle="--", alpha=0.7, zorder=0)
        ax.set_xlabel("Top-class Confidence (%)", fontsize=16)
        ax.set_ylabel("Success Rate (%)", fontsize=16)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_aspect("equal", "box")
        fig.tight_layout()
        return fig, ax


class CE:
    r"""`Top-label Calibration Error`_.

    See
    `CalibrationError <https://torchmetrics.readthedocs.io/en/stable/classification/calibration_error.html>`_
    for details. Our version of the metric is a wrapper around the original
    metric providing a plotting functionality.

    Reference:
        Naeini et al. "Obtaining well calibrated probabilities using Bayesian binning." In AAAI, 2015.
    """

    def __new__(  # type: ignore[misc]
        cls,
        task: Literal["binary", "multiclass"],
        n_bins: int = 10,
        norm: Literal["l1", "l2", "max"] = "l1",
        num_classes: int | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        task = ClassificationTaskNoMultilabel.from_str(task)
        kwargs.update(
            {
                "n_bins": n_bins,
                "norm": norm,
                "ignore_index": ignore_index,
                "validate_args": validate_args,
            }
        )
        if task == ClassificationTaskNoMultilabel.BINARY:
            return BinaryCE(**kwargs)
        if task == ClassificationTaskNoMultilabel.MULTICLASS:
            if not isinstance(num_classes, int):
                raise ValueError(
                    f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
                )
            return MulticlassCE(num_classes, **kwargs)
        raise ValueError(f"Not handled value: {task}")  # coverage: ignore


class AdaptiveCalibrationError(Metric):
    r"""`Adaptive Top-label Calibration Error`.

    See
    `CalibrationError <https://torchmetrics.readthedocs.io/en/stable/classification/calibration_error.html>`_
    for details of the original ECE. Instead of using fixed-length bins, this
    metric uses adaptive bins based on the confidence values. Each bin contains
    the same number of samples.

    Reference:
        Nixon et al. Measuring calibration in deep learning. In CVPRW, 2019.
    """

    is_differentiable: bool = False
    higher_is_better: bool = False
    full_state_update: bool = False

    confidences: list[torch.Tensor]
    accuracies: list[torch.Tensor]

    def __init__(self, n_bins: int = 15, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.nbins = n_bins
        self.add_state("confidences", [], dist_reduce_fx="cat")
        self.add_state("accuracies", [], dist_reduce_fx="cat")

    def histedges_equal(self, x):
        npt = len(x)
        return np.interp(
            np.linspace(0, npt, self.nbins + 1), np.arange(npt), np.sort(x)
        )

    def update(self, probs: torch.Tensor, targets: torch.Tensor) -> None:
        """Update metric states with predictions and targets."""
        confidences, preds = torch.max(probs, 1)
        accuracies = preds.eq(targets)
        self.confidences.append(confidences)
        self.accuracies.append(accuracies)

    def compute(self) -> torch.Tensor:
        """Compute metric."""
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)

        # Get edges
        bin_boundaries = np.histogram(
            a=confidences.cpu().detach(),
            bins=self.histedges_equal(confidences.cpu().detach()),
        )[1]

        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

        adaptive_ece = torch.zeros(1, device=confidences.device)
        for bin_lower, bin_upper in zip(
            self.bin_lowers, self.bin_uppers, strict=False
        ):
            in_bin = (confidences > bin_lower.item()) * (
                confidences < bin_upper.item()
            )
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                adaptive_ece += (
                    torch.abs(avg_confidence_in_bin - accuracy_in_bin)
                    * prop_in_bin
                )
        return adaptive_ece
