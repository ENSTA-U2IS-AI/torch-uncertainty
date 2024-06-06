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

from .adaptive_calibration_error import AdaptiveCalibrationError


def _ce_plot(self, ax: _AX_TYPE | None = None) -> _PLOT_OUT_TYPE:
    fig, ax = plt.subplots(figsize=(6, 6)) if ax is None else (None, ax)

    conf = dim_zero_cat(self.confidences)
    acc = dim_zero_cat(self.accuracies)
    bin_width = 1 / self.n_bins

    bin_ids = torch.round(
        torch.clamp(conf * self.n_bins, 1e-5, self.n_bins - 1 - 1e-5)
    )
    val, inverse, counts = bin_ids.unique(
        return_inverse=True, return_counts=True
    )
    counts = counts.float()
    val_oh = torch.nn.functional.one_hot(
        val.long(), num_classes=self.n_bins
    ).float()

    # add 1e-6 to avoid division NaNs
    values = (
        val_oh.T
        @ torch.sum(
            acc.unsqueeze(1) * torch.nn.functional.one_hot(inverse).float(),
            0,
        )
        / (val_oh.T @ counts + 1e-6)
    )

    plt.rc("axes", axisbelow=True)
    ax.hist(
        x=[bin_width * i * 100 for i in range(self.n_bins)],
        weights=values.cpu() * 100,
        bins=[bin_width * i * 100 for i in range(self.n_bins + 1)],
        alpha=0.7,
        linewidth=1,
        edgecolor="#0d559f",
        color="#1f77b4",
    )

    ax.plot([0, 100], [0, 100], "--", color="#0d559f")
    plt.grid(True, linestyle="--", alpha=0.7, zorder=0)
    ax.set_xlabel("Top-class Confidence (%)", fontsize=16)
    ax.set_ylabel("Success Rate (%)", fontsize=16)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_aspect("equal", "box")
    if fig is not None:
        fig.tight_layout()
    return fig, ax


# overwrite the plot method of the original metrics
BinaryCalibrationError.plot = _ce_plot
MulticlassCalibrationError.plot = _ce_plot


class CalibrationError:
    r"""Top-label Calibration Error.

    See
    `CalibrationError <https://torchmetrics.readthedocs.io/en/stable/classification/calibration_error.html>`_
    for details. Our version of the metric is a wrapper around the original
    metric providing a plotting functionality.

    Reference:
        Naeini et al. "Obtaining well calibrated probabilities using Bayesian
            binning." In AAAI, 2015.
    """

    def __new__(  # type: ignore[misc]
        cls,
        task: Literal["binary", "multiclass"],
        adaptive: bool = False,
        num_bins: int = 10,
        norm: Literal["l1", "l2", "max"] = "l1",
        num_classes: int | None = None,
        ignore_index: int | None = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> Metric:
        """Initialize task metric."""
        if adaptive:
            return AdaptiveCalibrationError(
                task=task,
                num_bins=num_bins,
                norm=norm,
                num_classes=num_classes,
                ignore_index=ignore_index,
                validate_args=validate_args,
                **kwargs,
            )
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
            return BinaryCalibrationError(**kwargs)
        #  task is ClassificationTaskNoMultilabel.MULTICLASS
        if not isinstance(num_classes, int):
            raise TypeError(
                f"`num_classes` is expected to be `int` but `{type(num_classes)} was passed.`"
            )
        return MulticlassCalibrationError(num_classes, **kwargs)
