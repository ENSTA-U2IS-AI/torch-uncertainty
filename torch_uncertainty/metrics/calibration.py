from typing import Any, Literal

import matplotlib.pyplot as plt
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


class CE:
    r"""`Top-label Calibration Error <https://arxiv.org/pdf/1909.10155.pdf>`_.

    See
    `CalibrationError <https://torchmetrics.readthedocs.io/en/stable/classification/calibration_error.html>`_
    for details. Our version of the metric is a wrapper around the original
    metric providing a plotting functionality.
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
