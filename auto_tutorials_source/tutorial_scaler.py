# -*- coding: utf-8 -*-
# fmt: off
# flake: noqa
"""
Improve Top-label Calibration with Temperature Scaling
======================================================

In this tutorial, we use torch-uncertainty to improve the calibration of the top-label predictions
to improve the reliability of the underlying neural network.

We also see how to use the datamodules outside any Lightning Trainer.

1. Loading the utilities
~~~~~~~~~~~~~~~~~~~~~~~~

In this tutorial, we will need:

- torch to download the pretrained model
- the Calibration Error metric to compute the ECE and evaluate the top-label calibration
- the CIFAR-100 datamodule to handle the data
- the Temperature Scaler to improve the top-label calibration
"""

import torch
from torchmetrics import CalibrationError

from torch_uncertainty.datamodules import CIFAR100DataModule
from torch_uncertainty.post_processing import TemperatureScaler

# %%
# 2. Downloading a Pre-trained Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To avoid training a model on CIFAR-100 from scratch, we will use here a model from https://github.com/chenyaofo/pytorch-cifar-models (thank you!)
# This can be done in a one liner:

model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet20", pretrained=True)

#%%
# 3. Setting up the Datamodule and Dataloader
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To get the dataloader from the datamodule, just call prepare_data, setup, and 
# extract the first element of the test dataloader list. There are more than one 
# element if `:attr:ood_detection` is True.

dm = CIFAR100DataModule(root="./data", ood_detection=False, batch_size=32)
dm.prepare_data()
dm.setup("test")

# Get the full test dataloader (unused in this tutorial)
dataloader = dm.test_dataloader()[0]

#%%
# 4. Iterating on the Dataloader and compute the ECE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We first split the original test set into a calibration set and a test set for proper evaluation.
#
# When computing the ECE, you need to provide the likelihoods associated with the inputs.
# To do this, just call PyTorch's softmax.

from torch.utils.data import DataLoader, random_split

# Split datasets
dataset = dm.test
cal_dataset, test_dataset = random_split(dataset, [1000, len(dataset)-1000])
cal_dataloader, test_dataloader = DataLoader(cal_dataset, batch_size=32), DataLoader(test_dataset, batch_size=32)

# Initialize the ECE
ece = CalibrationError(task="multiclass", num_classes=100)

# Iterate on the calibration dataloader
for sample, target in test_dataloader:
    logits = model(sample)
    ece.update(logits, target)

# Compute & print the calibration error
cal = ece.compute()

print(f"ECE before scaling - {cal*100:.3}%.")

#%%
# 5. Fittin the Scaler to Improve the Calibration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The TemperatureScaler has one parameter that can be used to temper the softmax.
# We minimize the tempered cross-entropy on a calibration set that we define here as
# a subset of the test set and containing 1000 data.

# Fit the scaler on the calibration dataset
scaler = TemperatureScaler()
scaler = scaler.fit(model=model, calib_loader=cal_dataloader)

#%%
# 6. Iterating again to compute the improved ECE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We create a wrapper of the original model and the scaler using torch.nn.Sequential.
# This is possible because the scaler is derived from nn.Module.
#
# Note that you will need to first reset the ECE metric to avoid mixing the scores of the previous and current iterations.

# Create the calibrated model
cal_model = torch.nn.Sequential(model, scaler)

# Reset the ECE
ece.reset()

# Iterate on the test dataloader
for sample, target in test_dataloader:
    logits = cal_model(sample)
    ece.update(logits.softmax(-1), target)

cal = ece.compute()

print(f"ECE after scaling - {cal*100:.3}%.")

# %%
# The top-label calibration should be improved.
#
# Notes
# -----
#
# Temperature scaling is very efficient when the calibration set is representative of the test set.
# In this case, we say that the calibration and test set are drawn from the same distribution.
# However, this may not be True in real-world cases where dataset shift could happen.

# %%
# References
# ----------
#
# - **Expected Calibration Error:** Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining Well Calibrated Probabilities Using Bayesian Binning. In `AAAI 2015 <https://arxiv.org/pdf/1411.0160.pdf>`_.
# - **Temperature Scaling:** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In `ICML 2017 <https://arxiv.org/pdf/1706.04599.pdf>`_.
