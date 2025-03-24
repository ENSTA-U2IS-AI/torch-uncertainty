"""
Out-of-distribution detection with TorchUncertainty
====================================================

This tutorial demonstrates how to perform OOD detection using
TorchUncertainty's ClassificationRoutine with a ResNet18 model trained on CIFAR-10,
evaluating its performance with SVHN as the OOD dataset.

We will:
- Set up the CIFAR-10 datamodule.
- Initialize and shortly train a ResNet18 model using the ClassificationRoutine.
- Evaluate the model's performance on both in-distribution and out-of-distribution data.
- Analyze uncertainty metrics for OOD detection.
"""

# %%
# Imports and Setup
# ------------------
#
# First, we need to import the necessary libraries and set up our environment.
# This includes importing PyTorch, TorchUncertainty components, and TorchUncertainty's Trainer (built on top of Lightning's),
# as well as two criteria for OOD detection, the Maximum Softmax Probability (MSP) and the Max Logit [1].
from torch import nn, optim

from torch_uncertainty import TUTrainer
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.models.resnet import resnet
from torch_uncertainty.routines.classification import ClassificationRoutine

# from torch_uncertainty.ood_criteria import MaxSoftmaxCriterion, MaxLogitCriterion

# %%
# Data Modules Setup
# ------------------
#
# TorchUncertainty provides convenient DataModules for standard datasets like CIFAR-10.
# DataModules handle data loading, preprocessing, and batching, simplifying the data pipeline. Each datamodule
# also include the corresponding out-of-distribution and distribution shift datasets, which are used by the routine.
# For CIFAR-10, the corresponding OOD-detection dataset is SVHN as used in the community.
# To enable OOD evaluation, activate the `eval_ood` flag as below.

datamodule = CIFAR10DataModule(root="./data", batch_size=512, num_workers=8, eval_ood=True)

# %%
# Model Initialization
# --------------------
#
# We will use the ResNet18 architecture, a widely adopted convolutional neural network known for its deep residual learning capabilities.
# The model is initialized with 10 output classes corresponding to the CIFAR-10 dataset categories.

# Initialize the ResNet18 model
model = resnet(arch=18, in_channels=3, num_classes=10)

# %%
# Define the Classification Routine
# ---------------------------------
#
# The `ClassificationRoutine` is one of the most crucial building blocks in TorchUncertainty.
# It streamlines the training and evaluation processes.
# It integrates the model, loss function, and optimizer into a cohesive routine compatible with PyTorch Lightning's Trainer.
# This abstraction simplifies the implementation of standard training loops and evaluation protocols.
# To come back to what matters in this tutorial, the routine also handles OOD detection. To enable it,
# activate the `eval_ood` flag.

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Initialize the ClassificationRoutine
routine = ClassificationRoutine(
    model=model, num_classes=10, loss=criterion, optim_recipe=optimizer, eval_ood=True
)

# %%
# Training the Model
# ------------------
#
# With the routine defined, we can now set up the Trainer and commence training.
# The Trainer handles the training loop, including epoch management, logging, and checkpointing.
# We specify the maximum number of epochs, the precision and the device to be used.

# Initialize the TUTrainer
trainer = TUTrainer(max_epochs=10, precision="16-mixed", accelerator="cuda", devices=1)

# Train the model for 10 epochs using the CIFAR-10 DataModule
trainer.fit(routine, datamodule=datamodule)
# %%
# Evaluating on In-Distribution and Out-of-distribution Data
# ----------------------------------------------------------
#
# After training, it's essential to evaluate the model's performance on the in-distribution test set.
# This assessment provides insights into the model's accuracy and reliability on data it has been trained on.

# Evaluate the model on the CIFAR-10 (IID) and SVHN (OOD) test sets
trainer.test(routine, datamodule=datamodule)

# %%
# Changing the OOD Criterion
# --------------------------
#
# To be added after merging


# %%
# References
# ----------
#
# [1]
