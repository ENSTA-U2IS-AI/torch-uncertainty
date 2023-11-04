import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.figure import Figure


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
