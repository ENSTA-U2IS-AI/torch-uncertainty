"""
Corrupting Images with TorchUncertainty to Benchmark Robustness
===============================================================

This tutorial shows the impact of the different corruption transforms available in the
TorchUncertainty library. These corruption transforms were first proposed in the paper
Benchmarking Neural Network Robustness to Common Corruptions and Perturbations
by Dan Hendrycks and Thomas Dietterich.

For this tutorial, we will only load the corruption transforms available in 
torch_uncertainty.transforms.corruption. We also need to load utilities from
torchvision and matplotlib.
"""
# %%
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop

import matplotlib.pyplot as plt
from PIL import Image
from urllib import request

urls = [
    "https://upload.wikimedia.org/wikipedia/commons/d/d9/Carduelis_tristis_-Michigan%2C_USA_-male-8.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/5/5d/Border_Collie_Blanca_y_Negra_Hembra_%28Belen%2C_Border_Collie_Los_Baganes%29.png",
    "https://upload.wikimedia.org/wikipedia/commons/f/f8/Birmakatze_Seal-Point.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/a/a9/Garranos_fight.jpg",
    "https://upload.wikimedia.org/wikipedia/commons/8/8b/Cottontail_Rabbit.jpg",
]

def download_img(url, i):
    request.urlretrieve(url, f"tmp_{i}.png")
    return Image.open(f"tmp_{i}.png").convert('RGB')

images_ds = [download_img(url, i) for i, url in enumerate(urls)]


def get_images(main_corruption, index: int = 0):
    """Create an image showing the 6 levels of corruption of a given transform."""
    images = []
    for severity in range(6):
        transforms = Compose(
            [Resize(256, antialias=True), CenterCrop(256), ToTensor(), main_corruption(severity), CenterCrop(224)]
        )
        images.append(transforms(images_ds[index]).permute(1, 2, 0).numpy())
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
from torch_uncertainty.transforms.corruption import (
    GaussianNoise,
    ShotNoise,
    ImpulseNoise,
)

show_images(
    [
        GaussianNoise,
        ShotNoise,
        ImpulseNoise,
    ]
)

# %%
# 2. Blur Corruptions
# ~~~~~~~~~~~~~~~~~~~~
from torch_uncertainty.transforms.corruption import (
    MotionBlur,
    GlassBlur,
    DefocusBlur,
    ZoomBlur,
)

show_images(
    [
        GlassBlur,
        MotionBlur,
        DefocusBlur,
        ZoomBlur,
    ]
)

# %%
# 3. Weather Corruptions
# ~~~~~~~~~~~~~~~~~~~~~~
from torch_uncertainty.transforms.corruption import (
    Frost,
    Snow,
    Fog,
)

show_images(
    [
        Fog,
        Frost,
        Snow,
    ]
)

# %%
# 4. Other Corruptions
# ~~~~~~~~~~~~~~~~~~~~

from torch_uncertainty.transforms.corruption import (
    Brightness, Contrast, Elastic, JPEGCompression, Pixelate)

show_images(
    [
        Brightness,
        Contrast,
        JPEGCompression,
        Pixelate,
        Elastic,
    ]
)

# %%
# 5. Unused Corruptions
# ~~~~~~~~~~~~~~~~~~~~~

# The following corruptions are not used in the paper Benchmarking Neural Network Robustness to Common Corruptions and Perturbations.

from torch_uncertainty.transforms.corruption import (
    GaussianBlur,
    SpeckleNoise,
    Saturation,
)

show_images(
    [
        GaussianBlur,
        SpeckleNoise,
        Saturation,
    ]
)

# %%
# Reference
# ---------
#
# - **Benchmarking Neural Network Robustness to Common Corruptions and Perturbations**, Dan Hendrycks and Thomas Dietterich. `ICLR 2019 <https://arxiv.org/pdf/1903.12261>`_.
