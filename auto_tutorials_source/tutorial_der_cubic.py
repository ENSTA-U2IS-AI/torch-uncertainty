"""
Deep Evidential Regression on a Toy Example
===========================================

This tutorial aims to provide an introductory overview of Deep Evidential Regression (DER) using a practical example. We demonstrate an application of DER by tackling the toy-problem of fitting :math:`y=x^3` using a Multi-Layer Perceptron (MLP) neural network model. The output layer of the MLP has four outputs, and is trained by minimizing the Normal Inverse-Gamma (NIG) loss function.

DER represents an evidential approach to quantifying uncertainty in neural network regression models. This method involves introducing prior distributions over the parameters of the Gaussian likelihood function. Then, the MLP model estimate the parameters of the evidential distribution.

Training a MLP with DER using TorchUncertainty models and PyTorch Lightning
---------------------------------------------------------------------------

In this part, we train a neural network, based on the model and routines already implemented in TU.

1. Loading the utilities
~~~~~~~~~~~~~~~~~~~~~~~~

To train a MLP with the NIG loss function using TorchUncertainty, we have to load the following utilities from TorchUncertainty:

- the cli handler: cli_main and argument parser: init_args
- the model: mlp, which lies in the torch_uncertainty.baselines.regression.mlp module.
- the regression training routine in the torch_uncertainty.routines.regression module.
- the evidential objective: the NIGLoss, which lies in the torch_uncertainty.losses file
- a dataset that generates samples from a noisy cubic function: Cubic, which lies in the torch_uncertainty.datasets.regression
"""

from pytorch_lightning import LightningDataModule
from torch_uncertainty import cli_main, init_args
from torch_uncertainty.baselines.regression.mlp import mlp
from torch_uncertainty.datasets.regression.toy import Cubic
from torch_uncertainty.losses import NIGLoss
from torch_uncertainty.routines.regression import RegressionSingle

# %%
# We also need to define an optimizer using torch.optim as well as the
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
# We use the Adam optimizer with the default learning rate of 0.001.


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
# In the following, we need to define the root of the logs, and to
# fake-parse the arguments needed for using the PyTorch Lightning Trainer. We
# also use the same synthetic regression task example as that used in the
# original DER paper.

root = Path(os.path.abspath(""))

# We mock the arguments for the trainer
with ArgvContext(
    "file.py",
    "--max_epochs",
    "50",
    "--enable_progress_bar",
    "False",
):
    args = init_args()
    args.use_cv = False

net_name = "logs/der-mlp-cubic"

# dataset
train_ds = Cubic(num_samples=1000)
val_ds = Cubic(num_samples=300)
test_ds = train_ds

# datamodule

datamodule = LightningDataModule.from_datasets(
    train_ds, val_dataset=val_ds, test_dataset=test_ds, batch_size=32
)
datamodule.training_task = "regression"

# model
model = mlp(in_features=1, num_outputs=4, hidden_dims=[64, 64])

# %%
# 4. The Loss and the Training Routine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Next, we need to define the loss to be used during training. To do this, we
# redefine the default parameters for the NIG loss using the partial
# function from functools. After that, we define the training routine using
# the regression training routine from torch_uncertainty.routines.regression. In
# this routine, we provide the model, the NIG loss, and the optimizer,
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
# Reference
# ---------
#
# - **Deep Evidential Regression:** Alexander Amini, Wilko Schwarting, Ava Soleimany, & Daniela Rus. `NeurIPS 2020 <https://arxiv.org/pdf/1910.02600>`_.
