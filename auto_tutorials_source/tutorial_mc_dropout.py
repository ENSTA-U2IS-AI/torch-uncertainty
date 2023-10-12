# -*- coding: utf-8 -*-
# flake: noqa
"""
Train a ResNet with Monte-Carlo Dropout
=======================================

In this tutorial, we'll train a ResNet classifier on the MNIST dataset using Monte-Carlo Dropout (MC Dropout), a computationally efficient Bayesian approximation method. To estimate the predictive mean and uncertainty (variance), we perform multiple forward passes through the network with dropout layers enabled in ``train`` mode.

For more information on Monte-Carlo Dropout, we refer the reader to the following resources:

- What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? `NeurIPS 2017 <https://browse.arxiv.org/pdf/1703.04977.pdf>`_
- Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning `PMLR 2016 <https://browse.arxiv.org/pdf/1506.02142.pdf>`_

Training a ResNet with MC Dropout using TorchUncertainty models and PyTorch Lightning
-------------------------------------------------------------------------------------

In this part, we train a ResNet with dropout layers, based on the model and routines already implemented in TU.

1. Loading the utilities
~~~~~~~~~~~~~~~~~~~~~~~~

First, we have to load the following utilities from TorchUncertainty:

- the cli handler: cli_main and argument parser: init_args
- the model: resnet18, which lies in the torch_uncertainty.models.resnet module
- the classification training routine in the torch_uncertainty.training.classification module
- the datamodule that handles dataloaders: MNISTDataModule, which lies in the torch_uncertainty.datamodule
- the optimizer wrapper in the torch_uncertainty.optimization_procedures module.
"""

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.datamodules import MNISTDataModule
from torch_uncertainty.baselines import ResNet
from torch_uncertainty.models.resnet import resnet18
from torch_uncertainty.routines.classification import ClassificationEnsemble
from torch_uncertainty.optimization_procedures import optim_cifar10_resnet18

# %%
# We will also need import the neural network utils withing `torch.nn`.
#
# We also import ArgvContext to avoid using the jupyter arguments as cli
# arguments, and therefore avoid errors.

import os
from functools import partial
from pathlib import Path

from torch import nn
from cli_test_helpers import ArgvContext

# %%
# 2. Creating the necessary variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the following, we will need to define the root of the datasets and the
# logs, and to fake-parse the arguments needed for using the PyTorch Lightning
# Trainer. We also create the datamodule that handles the MNIST dataset,
# dataloaders and transforms. Finally, we create the model using the
# blueprint from torch_uncertainty.models.
# 
# It is important to specify the arguments ``version`` as ``mc-dropout``,
# ``num_estimators`` and the ``dropout_rate`` to use Monte Carlo dropout.

root = Path(os.path.abspath(""))

# We mock the arguments for the trainer
with ArgvContext(
    "file.py",
    "--max_epochs",
    "1",
    "--enable_progress_bar",
    "False",
    "--version",
    "mc-dropout",
    "--dropout_rate",
    "0.4",
    "--num_estimators",
    "11",
):
    args = init_args(network=ResNet, datamodule=MNISTDataModule)

net_name = "mc-dropout-resnet18-mnist"

# datamodule
args.root = str(root / "data")
dm = MNISTDataModule(**vars(args))


model = resnet18(
    num_classes=dm.num_classes,
    in_channels=dm.num_channels,
    dropout_rate=args.dropout_rate,
    num_estimators=args.num_estimators,
)

# %%
# 3. The Loss and the Training Routine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is a classification problem, and we use CrossEntropyLoss as the likelihood.
# We define the training routine using the classification training routine from
# torch_uncertainty.training.classification. We provide the number of classes
# and channels, the optimizer wrapper, the dropout rate, and the number of
# forward passes to perform through the network, as well as all the default
# arguments.

from torch_uncertainty.transforms import RepeatTarget

format_batch_fn = RepeatTarget(num_repeats=args.num_estimators)

baseline = ClassificationEnsemble(
    num_classes=dm.num_classes,
    model=model,
    loss=nn.CrossEntropyLoss,
    optimization_procedure=optim_cifar10_resnet18,
    format_batch_fn=format_batch_fn,
    **vars(args),
)

# %%
# 5. Gathering Everything and Training the Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

results = cli_main(baseline, dm, root, net_name, args)

# %%
# 6. Testing the Model
# ~~~~~~~~~~~~~~~~~~~~
# Now that the model is trained, let's test it on MNIST

import matplotlib.pyplot as plt
import torch
import torchvision

import numpy as np

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


dataiter = iter(dm.val_dataloader())
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images[:4, ...]))
print("Ground truth: ", " ".join(f"{labels[j]}" for j in range(4)))

logits = model(images)
probs = torch.nn.functional.softmax(logits, dim=-1)

_, predicted = torch.max(probs, 1)

print("Predicted digits: ", " ".join(f"{predicted[j]}" for j in range(4)))
