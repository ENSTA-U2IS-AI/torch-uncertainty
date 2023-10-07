#!/usr/bin/env python
# coding: utf-8

"""
Deep Evidential Regression Tutorial
===================================

This tutorial aims to provide an introductory overview of Deep Evidential Regression (DER) using a practical example. We will demonstrate the application of DER by tackling the problem of fitting :math:`y=x^3` using a Multi-Layer Perceptron (MLP) neural network model. The output layer of the MLP would have four outputs, and will be trained by minimizing the Normal Inverse-Gamma (NIG) loss function.

DER represents a non-Bayesian approach to quantifying uncertainty in neural network regression models. This method involves introducing prior distributions over the parameters of the Gaussian likelihood function. Then, the MLP model estimate the parameters of the evidential distribution. 

Training a MLP with DER using TorchUncertainty models and PyTorch Lightning
---------------------------------------------------------------------------

In this part, we train a neural network, based on the model and routines already implemented in TU.

1. Loading the utilities
~~~~~~~~~~~~~~~~~~~~~~~~

To train a MLP with the NIG loss function using TorchUncertainty, we have to load the following utilities from TorchUncertainty:

- the cli handler: cli_main and argument parser: init_args
- the model: bayesian_lenet, which lies in the torch_uncertainty.baselines.regression.mlp module.
- the regression training routine in the torch_uncertainty.routines.regression module.
- the evidential objective: the NIGLoss, which lies in the torch_uncertainty.losses file
- the datamodule that handles dataloaders: CubicDataModule, which lies in the torch_uncertainty.datamodules
"""

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines.regression.mlp import mlp
from torch_uncertainty.datamodules.cubic_regression import CubicDataModule
from torch_uncertainty.losses import NIGLoss
from torch_uncertainty.routines.regression import RegressionSingle

# %%
# We will also need to define an optimizer using torch.optim as well as the
# neural network utils withing torch.nn, as well as the partial util to provide
# the modified default arguments for the NIG loss.
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
# We will use the Adam optimizer with the default learning rate of 0.001.

def optim_regression(
    model: nn.Module,
    learning_rate: float = 5e-4,
) -> dict:
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0,
    )
    return {
        "optimizer": optimizer,
    }

# %%
# 3. Creating the necessary variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the following, we will need to define the root of the logs, and to
# fake-parse the arguments needed for using the PyTorch Lightning Trainer. We
# also use the same synthetic regression task example as that used in the
# original DER paper.

root = Path(os.path.abspath(""))

# We mock the arguments for the trainer
with ArgvContext(
    "file.py",
    "--max_epochs",
    "100",
    "--batch_size",
    "32",
    "--enable_progress_bar",
    "False",
):
    args = init_args(datamodule=CubicDataModule)

net_name = "der-mlp-cubic"

# datamodule
datamodule = CubicDataModule(num_samples=1000, **vars(args))

# model
model = mlp(in_features=1, num_outputs=4, hidden_dims=[64, 64])

# %%
# 4. The Loss and the Training Routine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Next, we need to define the loss to be used during training. To do this, we
# will redefine the default parameters for the NIG loss using the partial
# function from functools. After that, we will define the training routine using
# the regression training routine from torch_uncertainty.routines.regression. In
# this routine, we will provide the model, the NIG loss, and the optimizer,
# along with the dist_estimation parameter, which refers to the number of
# distribution parameters, and all the default arguments.

loss = partial(
    NIGLoss,
    reg_weight=1e-2,
)

baseline = RegressionSingle(
    model=model,
    loss=loss,
    optimization_procedure=optim_regression,
    dist_estimation=4,
    **vars(args),
)

# %%
# 5. Gathering Everything and Training the Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

results = cli_main(baseline, datamodule, root, net_name, args)

# %%
# 6. Testing the Model
# ~~~~~~~~~~~~~~~~~~~~

import matplotlib.pyplot as plt
from torch.nn import functional as F

with torch.no_grad():
    x = torch.linspace(-7, 7, 1000).unsqueeze(-1)

    logits = model(x)
    means, v, alpha, beta = logits.split(1, dim=-1)

    v = F.softplus(v)
    alpha = 1 + F.softplus(alpha)
    beta = F.softplus(beta)

    vars = torch.sqrt(beta / (v * (alpha - 1)))

    means.squeeze_(1)
    vars.squeeze_(1)
    x.squeeze_(1)

fig, ax = plt.subplots(1, 1)
ax.plot(x, x**3, "--r", label="ground truth", zorder=3)
ax.plot(x, means, "-k", label="predictions")
for k in torch.linspace(0, 4, 4):
    ax.fill_between(
        x,
        means - k * vars,
        means + k * vars,
        linewidth=0,
        alpha=0.3,
        edgecolor=None,
        facecolor="blue",
        label="epistemic uncertainty" if not k else None,
    )

plt.gca().set_ylim(-150, 150)
plt.gca().set_xlim(-7, 7)
plt.legend(loc="upper left")
plt.grid()

# %%
# References
# ----------
#
# - **Deep Evidential Regression:** Alexander Amini, Wilko Schwarting, Ava Soleimany, & Daniela Rus (2022). Deep Evidential Regression `NeurIPS 2022 <https://arxiv.org/pdf/1910.02600>`_
