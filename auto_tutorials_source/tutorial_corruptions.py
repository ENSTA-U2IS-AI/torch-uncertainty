"""
Image Corruptions
=================

This tutorial shows the impact of the different corruptions available in the
TorchUncertainty library. These corruptions were first proposed in the paper
Benchmarking Neural Network Robustness to Common Corruptions and Perturbations
by Dan Hendrycks and Thomas Dietterich.

For this tutorial, we will only load the corruption transforms available in 
torch_uncertainty.transforms.corruptions. We also need to load utilities from
torchvision and matplotlib.
"""
import torch
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Resize

from torchvision.utils import make_grid
import matplotlib.pyplot as plt

ds = CIFAR10("./data", train=False, download=True)

def get_images(main_transform, severity):
    ds_transforms = Compose([ToTensor(), main_transform(severity), Resize(256)])
    ds = CIFAR10("./data", train=False, download=False, transform=ds_transforms)
    return make_grid([ds[i][0] for i in range(6)]).permute(1, 2, 0)

def show_images(transform):
    print("Original Images")
    with torch.no_grad():
        plt.axis('off')
        plt.imshow(get_images(transform, 0))
        plt.show()

    for severity in range(1, 6):
        print(f"Severity {severity}")
        with torch.no_grad():
            plt.axis('off')
            plt.imshow(get_images(transform, severity))
            plt.show()

# %%
# 1. Gaussian Noise
# ~~~~~~~~~~~~~~~~~
from torch_uncertainty.transforms.corruptions import GaussianNoise

show_images(GaussianNoise)

# %%
# 2. Shot Noise
# ~~~~~~~~~~~~~
from torch_uncertainty.transforms.corruptions import ShotNoise

show_images(ShotNoise)

# %%
# 3. Impulse Noise
# ~~~~~~~~~~~~~~~~
from torch_uncertainty.transforms.corruptions import ImpulseNoise

show_images(ImpulseNoise)

# %%
# 4. Speckle Noise
# ~~~~~~~~~~~~~~~~
from torch_uncertainty.transforms.corruptions import SpeckleNoise

show_images(SpeckleNoise)

# %%
# 5. Gaussian Blur
# ~~~~~~~~~~~~~~~~
from torch_uncertainty.transforms.corruptions import GaussianBlur

show_images(GaussianBlur)

# %%
# 6. Glass Blur
# ~~~~~~~~~~~~~
from torch_uncertainty.transforms.corruptions import GlassBlur

show_images(GlassBlur)

# %%
# 7. Defocus Blur
# ~~~~~~~~~~~~~~~

from torch_uncertainty.transforms.corruptions import DefocusBlur

show_images(DefocusBlur)

#%%
# 8. JPEG Compression
# ~~~~~~~~~~~~~~
from torch_uncertainty.transforms.corruptions import JPEGCompression

show_images(JPEGCompression)

#%%
# 9. Pixelate
# ~~~~~~~~~~~
from torch_uncertainty.transforms.corruptions import Pixelate

show_images(Pixelate)

#%% 
# 10. Frost
# ~~~~~~~~
from torch_uncertainty.transforms.corruptions import Frost

show_images(Frost)

# %%
# Reference
# ---------
#
# - **Benchmarking Neural Network Robustness to Common Corruptions and Perturbations**, Dan Hendrycks and Thomas Dietterich. `ICLR 2019 <https://arxiv.org/pdf/1903.12261>`_.
