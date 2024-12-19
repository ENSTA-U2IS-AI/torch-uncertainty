"""
Deep Probabilistic Regression
=============================

This tutorial aims to provide an overview of some utilities in TorchUncertainty for probabilistic regression.

Building a MLP for Probabilistic Regression using TorchUncertainty distribution layers
--------------------------------------------------------------------------------------

In this section we cover the building of a very simple MLP outputting Normal distribution parameters.

1. Loading the utilities
~~~~~~~~~~~~~~~~~~~~~~~~

We disable some logging and warnings to keep the output clean.
"""
# %%
import torch
from torch import nn

import logging
logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)

import warnings
warnings.filterwarnings("ignore")

# %%
# 2. Building the MLP model
# ~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To create a MLP model estimating a Normal distribution, we use the NormalLinear layer.
# This layer is a wrapper around the nn.Linear layer, which outputs the location and scale of a Normal distribution.
# Note that any other distribution layer from TU can be used in the same way.
from torch_uncertainty.layers.distributions import NormalLinear


class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 50)
        self.fc2 = NormalLinear(
            base_layer=nn.Linear,
            event_dim=out_features,
            in_features=50,
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# %%
# 3. Setting up the data
# ~~~~~~~~~~~~~~~~~~~~~~
#
# We use the UCI Kin8nm dataset, which is a regression dataset with 8 features and 8192 samples.
from torch_uncertainty.datamodules import UCIRegressionDataModule

# datamodule
datamodule = UCIRegressionDataModule(
    root="data",
    batch_size=32,
    dataset_name="kin8nm",
)

# %%
# 4. Setting up the model and trainer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from torch_uncertainty import TUTrainer

trainer = TUTrainer(
    accelerator="cpu",
    max_epochs=5,
    enable_progress_bar=False,
)

model = MLP(in_features=8, out_features=1)


# %%
# 5. The Loss, the Optimizer and the Training Routine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We use the DistributionNLLLoss to compute the negative log-likelihood of the Normal distribution.
# Note that this loss can be used with any Distribution from torch.distributions.
# For the optimizer, we use the Adam optimizer with a learning rate of 5e-3.
# Finally, we create a RegressionRoutine to train the model. We indicate that the output dimension is 1 and the distribution family is "normal".

from torch_uncertainty.losses import DistributionNLLLoss
from torch_uncertainty.routines import RegressionRoutine

loss = DistributionNLLLoss()

def optim_regression(
    model: nn.Module,
    learning_rate: float = 5e-3,
):
    return torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0,
    )

routine = RegressionRoutine(
    output_dim=1,
    model=model,
    loss=loss,
    optim_recipe=optim_regression(model),
    dist_family="normal",
)


# %%
# 6. Training the model
# ~~~~~~~~~~~~~~~~~~~~~~

trainer.fit(model=routine, datamodule=datamodule)
results = trainer.test(model=routine, datamodule=datamodule)

# %%
# 7. Benchmarking different distributions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Our MLP model assumes a Normal distribution as the output. However, we could be interested in comparing the performance of different distributions.
# TorchUncertainty provides a simple way to do this using the get_dist_linear_layer() function.
# Let us rewrite the MLP model to use it.

from torch_uncertainty.layers.distributions import get_dist_linear_layer

class MLP(nn.Module):
    def __init__(self, in_features: int, out_features: int, dist_family: str):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 50)
        dist_layer = get_dist_linear_layer(dist_family)
        self.fc2 = dist_layer(
            base_layer=nn.Linear,
            event_dim=out_features,
            in_features=50,
        )

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# %%
# We can now train the model with different distributions.
# Let us train the model with a Normal, Laplace, Student's t, and Cauchy distribution.
# Note that we use the mode as the point-wise estimate of the distribution as the mean
# is not defined for the Cauchy distribution.
for dist_family in ["normal", "laplace", "student", "cauchy"]:
    print("#" * 50)
    print(f">>> Training with {dist_family} distribution")
    print("#" * 50)
    trainer = TUTrainer(
        accelerator="cpu",
        max_epochs=10,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    model = MLP(in_features=8, out_features=1, dist_family=dist_family)
    routine = RegressionRoutine(
        output_dim=1,
        model=model,
        loss=loss,
        optim_recipe=optim_regression(model),
        dist_family=dist_family,
        dist_estimate="mode",
    )
    trainer.fit(model=routine, datamodule=datamodule)
    trainer.test(model=routine, datamodule=datamodule)
