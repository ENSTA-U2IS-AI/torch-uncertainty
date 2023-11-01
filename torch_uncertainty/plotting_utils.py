import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


class CalibrationPlot:
    """A class for plotting calibration figures for classification models.

    Args:
        mode (str, optional): The mode of the calibration plot. One of
            "top_label" (default).
        adaptive (bool, optional): Whether to use adaptive binning. Defaults to
            ``False``.
        num_bins (int, optional): The number of bins. Defaults to ``10``.
        figsize (Tuple, optional): The figure size. Defaults to ``(5, 5)``.

    Raises:
        NotImplementedError: If ``mode`` is not "top_label".
        NotImplementedError: If ``adaptive`` is ``True``.
        TypeError: If ``num_bins`` is not an ``int``.
        ValueError: If ``num_bins`` is not strictly positive.
    """

    def __init__(
        self,
        mode: str = "top_label",
        adaptive: bool = False,
        num_bins: int = 10,
        figsize: tuple = (5, 5),
    ) -> None:
        if mode != "top_label":
            raise NotImplementedError(f"Mode {mode} is not yet implemented.")

        if adaptive:
            raise NotImplementedError(
                "Adaptive binning is not yet implemented."
            )

        if not isinstance(num_bins, int):
            raise TypeError(f"num_bins should be int. Got {type(num_bins)}.")
        if num_bins < 1:
            raise ValueError(
                f"num_bins should be strictly positive. Got {num_bins}."
            )

        self.num_bins = num_bins
        self.bin_width = 1 / num_bins

        self.figsize = figsize

        self.conf = []
        self.acc = []

    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        """Update the calibration plot with new predictions and targets.

        Args:
            preds (torch.Tensor): The prediction likelihoods (<1).
            targets (torch.Tensor): The targets.
        """
        if preds.ndim == 1:  # binary classification
            self.conf.append(preds)
        else:
            self.conf.append(preds.max(-1).values.cpu())

        if preds.ndim == 1:  # binary classification
            self.acc.append((preds.round() == targets).cpu())
        else:
            self.acc.append((preds.argmax(-1) == targets).cpu())

    def compute(self) -> tuple[Figure, Axes]:
        """Compute and plot the calibration figure.

        Returns:
            Tuple[Figure, Axes]: The figure and axes of the plot.
        """
        confidence = torch.cat(self.conf)
        acc = torch.cat(self.acc)

        bin_ids = torch.round(
            torch.clamp(
                confidence * self.num_bins, 1e-5, self.num_bins - 1 - 1e-5
            )
        )
        val, inverse, counts = bin_ids.unique(
            return_inverse=True, return_counts=True
        )
        val_oh = torch.nn.functional.one_hot(
            val.long(), num_classes=self.num_bins
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
        fig, ax = plt.subplots(1, figsize=self.figsize)
        ax.hist(
            x=[self.bin_width * i * 100 for i in range(self.num_bins)],
            weights=values * 100,
            bins=[self.bin_width * i * 100 for i in range(self.num_bins + 1)],
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

    def __call__(
        self, preds: torch.Tensor, targets: torch.Tensor
    ) -> tuple[Figure, Axes]:
        """Update, compute, and plot the calibration plot.

        Args:
            preds (torch.Tensor): The prediction likelihoods (<1).
            targets (torch.Tensor): The targets.

        Returns:
            Tuple[Figure, Axes]: The figure and axes of the plot.
        """
        self.update(preds, targets)
        return self.compute()


def plot_hist(
    conf: list[torch.Tensor],
    bins: int = 20,
    title: str = "Histogram with 'auto' bins",
    dpi: int = 60,
) -> tuple[Figure, Axes]:
    """Plot a confidence histogram.

    Args:
        conf (Any): The confidence values.
        bins (int, optional): The number of bins. Defaults to 20.
        title (str, optional): The title of the plot. Defaults to "Histogram
            with 'auto' bins".
        dpi (int, optional): The dpi of the plot. Defaults to 60.

    Returns:
        Tuple[Figure, Axes]: The figure and axes of the plot.
    """
    plt.rc("axes", axisbelow=True)
    fig, ax = plt.subplots(1, figsize=(7, 5), dpi=dpi)
    for i in [1, 0]:
        ax.hist(
            conf[i],
            bins=bins,
            density=True,
            label=["In-distribution", "Out-of-Distribution"][i],
            alpha=0.4,
            linewidth=1,
            edgecolor=["#0d559f", "#d45f00"][i],
            color=["#1f77b4", "#ff7f0e"][i],
        )

    ax.set_title(title)
    plt.grid(True, linestyle="--", alpha=0.7, zorder=0)
    plt.legend()
    fig.tight_layout()
    return fig, ax
