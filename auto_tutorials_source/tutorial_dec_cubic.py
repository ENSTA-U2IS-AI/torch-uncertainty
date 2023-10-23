#!/usr/bin/env python
# coding: utf-8

"""
Deep Evidential Classification on a Toy Example
===========================================

This tutorial aims to provide an introductory overview of Deep Evidential Classification (DEC) using a practical example. We demonstrate an application of DEC by tackling the toy-problem of fitting the MNIST dataset using a Multi-Layer Perceptron (MLP) neural network model. The output of the MLP is modeled as a Dirichlet distribution. The MLP is trained by minimizing the DEC loss function, composed of a Bayesian risk square error loss and a regularization term based on KL Divergence.

DEC represents an evidential approach to quantifying uncertainty in neural network classification models. This method involves introducing prior distributions over the parameters of the Categorical likelihood function. Then, the MLP model estimates the parameters of the evidential distribution.

Training a LeNet with DEC using TorchUncertainty models and PyTorch Lightning
---------------------------------------------------------------------------

In this part, we train a neural network, based on the model and routines already implemented in TU.

1. Loading the utilities
~~~~~~~~~~~~~~~~~~~~~~~~

To train a LeNet with the DEC loss function using TorchUncertainty, we have to load the following utilities from TorchUncertainty:

- the cli handler: cli_main and argument parser: init_args
- the model: LeNet, which lies in torch_uncertainty.models
- the classification training routine in the torch_uncertainty.training.classification module
- the evidential objective: the DECLoss, which lies in the torch_uncertainty.losses file
- the datamodule that handles dataloaders: MNISTDataModule, which lies in the torch_uncertainty.datamodule
- the optimizer wrapper in the torch_uncertainty.optimization_procedures module.
"""

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.datamodules import MNISTDataModule
from torch_uncertainty.losses import DECLoss
from torch_uncertainty.baselines import ResNet
from torch_uncertainty.models.lenet import lenet
from torch_uncertainty.routines.classification import ClassificationSingle


# %%
# We also need to define an optimizer using torch.optim as well as the
# neural network utils withing torch.nn, as well as the partial util to provide
# the modified default arguments for the DEC loss.
#
# We also import ArgvContext to avoid using the jupyter arguments as cli
# arguments, and therefore avoid errors.

import os
from functools import partial
from pathlib import Path

import torch
from cli_test_helpers import ArgvContext
from torch import nn, optim


# %%
# 2. Creating the Optimizer Wrapper
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We follow the official implementation in DEC, use the Adam optimizer
# with the default learning rate of 0.001 and a scheduler for the learning rate
def optim_lenet(model: nn.Module) -> dict:
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
        weight_decay=0.005
    )
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    return {"optimizer": optimizer, "lr_scheduler": exp_lr_scheduler}


# %%
# 3. Creating the necessary variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the following, we need to define the root of the logs, and to
# fake-parse the arguments needed for using the PyTorch Lightning Trainer. We
# also use the same MNIST classification example as that used in the
# original DEC paper.
root = Path(os.path.abspath(""))

# We mock the arguments for the trainer
with ArgvContext(
    "file.py",
    "--max_epochs",
    "25",
    "--enable_progress_bar",
    "True",
    "--version",
    "vanilla",
):
    args = init_args(network=ResNet, datamodule=MNISTDataModule)

net_name = "dec-lenet-mnist"

# datamodule
args.root = str(root / "data")
dm = MNISTDataModule(**vars(args))


model = lenet(
    in_channels=dm.num_channels,
    num_classes=dm.num_classes,
)

# %%
# 4. The Loss and the Training Routine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Next, we need to define the loss to be used during training. To do this, we
# redefine the default parameters for the DEC loss using the partial
# function from functools. After that, we define the training routine using
# the single classification model training routine from 
# torch_uncertainty.routines.classification.ClassificationSingle. 
# In this routine, we provide the model, the DEC loss, the optimizer,
# and all the default arguments.

loss = partial(
    DECLoss,
    reg_weight=1e-2,
)

baseline = ClassificationSingle(
    model=model,
    num_classes=dm.num_classes,
    in_channels=dm.num_channels,
    loss=loss,
    optimization_procedure=optim_lenet,
    **vars(args),
)

# %%
# 5. Gathering Everything and Training the Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

results = cli_main(baseline, dm, root, net_name, args)

# %%
# 6. Testing the Model
# ~~~~~~~~~~~~~~~~~~~~
# Now that the model is trained, let's test it on MNIST.

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms.functional as F

import numpy as np


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(dm.val_dataloader())
images, labels = next(dataiter)

with torch.no_grad():
    baseline.eval()

    # print original images
    imshow(torchvision.utils.make_grid(images[:4, ...]))
    print("Ground truth: ", " ".join(f"{labels[j]}" for j in range(4)))
    evidence = baseline(images)
    alpha = torch.relu(evidence) + 1
    strength = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha/strength
    entropy = -1 * torch.sum(probs * torch.log(probs), dim = 1, keepdim=True)
    for j in range(4):
        predicted = torch.argmax(probs[j, :])
        print(
            f"Predicted digits for the image {j}: {predicted} with strength {strength[j,0]} and entropy {entropy[j,0]}."
        )
    
    # rotate the images by 45 degrees
    rotated_images = F.rotate(images, 45)
    # print rotated images
    imshow(torchvision.utils.make_grid(rotated_images[:4, ...]))
    print("Ground truth: ", " ".join(f"{labels[j]}" for j in range(4)))
    
    evidence = baseline(rotated_images)
    alpha = torch.relu(evidence) + 1
    strength = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha/strength
    entropy = -1 * torch.sum(probs * torch.log(probs), dim = 1, keepdim=True)
    for j in range(4):
        predicted = torch.argmax(probs[j, :])
        print(
            f"Predicted digits for the image {j}: {predicted} with strength {strength[j,0]} and entropy {entropy[j,0]}."
        )

    # rotate the images by 90 degrees
    rotated_images = F.rotate(images, 90)
    # print rotated images
    imshow(torchvision.utils.make_grid(rotated_images[:4, ...]))
    print("Ground truth: ", " ".join(f"{labels[j]}" for j in range(4)))
    
    evidence = baseline(rotated_images)
    alpha = torch.relu(evidence) + 1
    strength = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha/strength
    entropy = -1 * torch.sum(probs * torch.log(probs), dim = 1, keepdim=True)
    for j in range(4):
        predicted = torch.argmax(probs[j, :])
        print(
            f"Predicted digits for the image {j}: {predicted} with strength {strength[j,0]} and entropy {entropy[j,0]}."
        )


# %%
# References
# ----------
#
# - **Deep Evidential Classification:** Murat Sensoy, Lance Kaplan, & Melih Kandemir (2018). Evidential Deep Learning to Quantify Classification Uncertainty `NeurIPS 2018 <https://arxiv.org/pdf/1806.01768>`_
