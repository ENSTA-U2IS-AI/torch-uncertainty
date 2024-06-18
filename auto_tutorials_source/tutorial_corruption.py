"""
Corrupting Images with TorchUncertainty to Benchmark Robustness
===============================================================

This tutorial shows the impact of the different corruptions available in the
TorchUncertainty library. These corruptions were first proposed in the paper
Benchmarking Neural Network Robustness to Common Corruptions and Perturbations
by Dan Hendrycks and Thomas Dietterich.

For this tutorial, we will only load the corruption transforms available in 
torch_uncertainty.transforms.corruptions. We also need to load utilities from
torchvision and matplotlib.
"""

from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Resize

import matplotlib.pyplot as plt

ds = CIFAR10("./data", train=False, download=True)


def get_images(main_corruption, index: int = 0):
    """Create an image showing the 6 levels of corruption of a given transform."""
    images = []
    for severity in range(6):
        ds_transforms = Compose(
            [ToTensor(), main_corruption(severity), Resize(256, antialias=True)]
        )
        ds = CIFAR10("./data", train=False, download=False, transform=ds_transforms)
        images.append(ds[index][0].permute(1, 2, 0).numpy())
    return images


def show_images(transforms):
    """Show the effect of all given transforms."""
    num_corruptions = len(transforms)
    _, ax = plt.subplots(num_corruptions, 6, figsize=(10, int(1.5 * num_corruptions)))
    for i, transform in enumerate(transforms):
        images = get_images(transform, index=i)
        ax[i][0].text(
            -0.1,
            0.5,
            transform.__name__,
            transform=ax[i][0].transAxes,
            rotation="vertical",
            horizontalalignment="right",
            verticalalignment="center",
            fontsize=12,
        )
        for j in range(6):
            ax[i][j].imshow(images[j])
            if i == 0 and j == 0:
                ax[i][j].set_title("Original")
            elif i == 0:
                ax[i][j].set_title(f"Severity {j}")
            ax[i][j].axis("off")
    plt.show()


# %%
# 1. Noise Corruptions
# ~~~~~~~~~~~~~~~~~~~~
from torch_uncertainty.transforms.corruptions import (
    GaussianNoise,
    ShotNoise,
    ImpulseNoise,
    SpeckleNoise,
)

show_images(
    [
        GaussianNoise,
        ShotNoise,
        ImpulseNoise,
        SpeckleNoise,
    ]
)

# %%
# 2. Blur Corruptions
# ~~~~~~~~~~~~~~~~~~~~
from torch_uncertainty.transforms.corruptions import (
    GaussianBlur,
    GlassBlur,
    DefocusBlur,
)

show_images(
    [
        GaussianBlur,
        GlassBlur,
        DefocusBlur,
    ]
)

# %%
# 3. Other Corruptions
# ~~~~~~~~~~~~~~~~~~~~
from torch_uncertainty.transforms.corruptions import (
    JPEGCompression,
    Pixelate,
    Frost,
)

show_images(
    [
        JPEGCompression,
        Pixelate,
        Frost,
    ]
)

# %%
# Reference
# ---------
#
# - **Benchmarking Neural Network Robustness to Common Corruptions and Perturbations**, Dan Hendrycks and Thomas Dietterich. `ICLR 2019 <https://arxiv.org/pdf/1903.12261>`_.
