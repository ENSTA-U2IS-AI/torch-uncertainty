"""
Image Corruptions
=================

This tutorial shows the impact of the different corruptions available in the
TorchUncertainty library. These corruptions were first proposed in the paper
Benchmarking Neural Network Robustness to Common Corruptions and Perturbations
by Dan Hendrycks and Thomas Dietterich.

For this tutorial, we will only load the corruption transforms available in 
torch_uncertainty.transforms.corruptions. We also need to load utilities from
torchvision and matplotlib
"""
import torch
from torchvision.datasets import CIFAR10, MNIST
from torchvision.transforms import Compose, ToTensor, Resize

from torchvision.utils import make_grid
import matplotlib.pyplot as plt
plt.axis('off')
ds = CIFAR10("./data", train=False, download=True)

def get_images(main_transform, severity):
    ds_transforms = Compose([ToTensor(), main_transform(severity), Resize(256)])
    ds = CIFAR10("./data", train=False, download=False, transform=ds_transforms)
    return [ds[i][0] for i in range(6)]

# %%
# 1. Gaussian Noise
# ~~~~~~~~~~~~~~~~~
from torch_uncertainty.transforms.corruptions import GaussianNoise

print("Original Images")
with torch.no_grad():
    plt.axis('off')
    plt.imshow(make_grid(get_images(GaussianNoise, 0)).permute(1, 2, 0))
    plt.show()

for severity in range(1, 6):
    print(f"Severity {severity}")
    with torch.no_grad():
        plt.axis('off')
        plt.imshow(make_grid(get_images(GaussianNoise, severity)).permute(1, 2, 0))
        plt.show()

# %%
# 2. Shot Noise
# ~~~~~~~~~~~~~
from torch_uncertainty.transforms.corruptions import ShotNoise

print("Original Images")
with torch.no_grad():
    plt.axis('off')
    plt.imshow(make_grid(get_images(ShotNoise, 0)).permute(1, 2, 0))
    plt.show()

for severity in range(1, 6):
    print(f"Severity {severity}")
    with torch.no_grad():
        plt.axis('off')
        plt.imshow(make_grid(get_images(ShotNoise, severity)).permute(1, 2, 0))
        plt.show()

# %%
# 3. Impulse Noise
# ~~~~~~~~~~~~~~~~
from torch_uncertainty.transforms.corruptions import ImpulseNoise

print("Original Images")
with torch.no_grad():
    plt.axis('off')
    plt.imshow(make_grid(get_images(ImpulseNoise, 0)).permute(1, 2, 0))
    plt.show()

for severity in range(1, 6):
    print(f"Severity {severity}")
    with torch.no_grad():
        plt.axis('off')
        plt.imshow(make_grid(get_images(ImpulseNoise, severity)).permute(1, 2, 0))
        plt.show()

# %%
# 4. Gaussian Blur
# ~~~~~~~~~~~~~~~~
from torch_uncertainty.transforms.corruptions import GaussianBlur

print("Original Images")
with torch.no_grad():
    plt.axis('off')
    plt.imshow(make_grid(get_images(GaussianBlur, 0)).permute(1, 2, 0))
    plt.show()

for severity in range(1, 6):
    print(f"Severity {severity}")
    with torch.no_grad():
        plt.axis('off')
        plt.imshow(make_grid(get_images(GaussianBlur, severity)).permute(1, 2, 0))
        plt.show()


# %%
# 5. Glass Blur
# ~~~~~~~~~~~~~
from torch_uncertainty.transforms.corruptions import GlassBlur

print("Original Images")
with torch.no_grad():
    plt.axis('off')
    plt.imshow(make_grid(get_images(GlassBlur, 0)).permute(1, 2, 0))
    plt.show()

for severity in range(1, 6):
    print(f"Severity {severity}")
    with torch.no_grad():
        plt.axis('off')
        plt.imshow(make_grid(get_images(GlassBlur, severity)).permute(1, 2, 0))
        plt.show()


# %%
# 6. Defocus Blur
# ~~~~~~~~~~~~~~~

from torch_uncertainty.transforms.corruptions import DefocusBlur

print("Original Images")
with torch.no_grad():
    plt.axis('off')
    plt.imshow(make_grid(get_images(DefocusBlur, 0)).permute(1, 2, 0))
    plt.show()

for severity in range(1, 6):
    print(f"Severity {severity}")
    with torch.no_grad():
        plt.axis('off')
        plt.imshow(make_grid(get_images(DefocusBlur, severity)).permute(1, 2, 0))
        plt.show()

# %%
# Reference
# ---------
#
# - **Benchmarking Neural Network Robustness to Common Corruptions and Perturbations**, Dan Hendrycks and Thomas Dietterich. `ICLR 2019 <https://arxiv.org/pdf/1903.12261>`_.
