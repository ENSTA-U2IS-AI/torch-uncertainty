"""
Improve Top-label Calibration with Temperature Scaling
======================================================

In this tutorial, we use *TorchUncertainty* to improve the calibration
of the top-label predictions and the reliability of the underlying neural network.

This tutorial provides extensive details on how to use the TemperatureScaler
class, however, this is done automatically in the classification routine when setting
the `calibration_set` to val or test.

Through this tutorial, we also see how to use the datamodules outside any Lightning trainers,
and how to use TorchUncertainty's models.

1. Loading the Utilities
~~~~~~~~~~~~~~~~~~~~~~~~

In this tutorial, we will need:

- TorchUncertainty's Calibration Error metric to compute to evaluate the top-label calibration with ECE and plot the reliability diagrams
- the CIFAR-100 datamodule to handle the data
- a ResNet 18 as starting model
- the temperature scaler to improve the top-label calibration
- a utility function to download HF models easily

If you use the classification routine, the plots will be automatically available in the tensorboard logs if you use the `log_plots` flag.
"""
# %%
from torch_uncertainty.datamodules import CIFAR100DataModule
from torch_uncertainty.metrics import CalibrationError
from torch_uncertainty.models.resnet import resnet
from torch_uncertainty.post_processing import TemperatureScaler
from torch_uncertainty.utils import load_hf

# %%
# 2. Loading a model from TorchUncertainty's HF
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To avoid training a model on CIFAR-100 from scratch, we load a model from Hugging Face.
# This can be done in a one liner:

# Build the model
model = resnet(in_channels=3, num_classes=100, arch=18, style="cifar", conv_bias=False)

# Download the weights (the config is not used here)
weights, config = load_hf("resnet18_c100")

# Load the weights in the pre-built model
model.load_state_dict(weights)

# %%
# 3. Setting up the Datamodule and Dataloaders
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# To get the dataloader from the datamodule, just call prepare_data, setup, and
# extract the first element of the test dataloader list. There are more than one
# element if eval_ood is True: the dataloader of in-distribution data and the dataloader
# of out-of-distribution data. Otherwise, it is a list of 1 element.

dm = CIFAR100DataModule(root="./data", eval_ood=False, batch_size=32)
dm.prepare_data()
dm.setup("test")

# Get the full test dataloader (unused in this tutorial)
dataloader = dm.test_dataloader()[0]

# %%
# 4. Iterating on the Dataloader and Computing the ECE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We first split the original test set into a calibration set and a test set for proper evaluation.
#
# When computing the ECE, you need to provide the likelihoods associated with the inputs.
# To do this, just call PyTorch's softmax.
#
# To avoid lengthy computations (without GPU), we restrict the calibration computation to a subset
# of the test set.

from torch.utils.data import DataLoader, random_split

# Split datasets
dataset = dm.test
cal_dataset, test_dataset, other = random_split(
    dataset, [1000, 1000, len(dataset) - 2000]
)
test_dataloader = DataLoader(test_dataset, batch_size=32)

# Initialize the ECE
ece = CalibrationError(task="multiclass", num_classes=100)

# Iterate on the calibration dataloader
for sample, target in test_dataloader:
    logits = model(sample)
    probs = logits.softmax(-1)
    ece.update(probs, target)

# Compute & print the calibration error
print(f"ECE before scaling - {ece.compute():.3%}.")

# %%
# We also compute and plot the top-label calibration figure. We see that the
# model is not well calibrated.
fig, ax = ece.plot()
fig.show()

# %%
# 5. Fitting the Scaler to Improve the Calibration
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# The TemperatureScaler has one parameter that can be used to temper the softmax.
# We minimize the tempered cross-entropy on a calibration set that we define here as
# a subset of the test set and containing 1000 data. Look at the code run by TemperatureScaler
# `fit` method for more details.

# Fit the scaler on the calibration dataset
scaled_model = TemperatureScaler(model=model)
scaled_model.fit(calibration_set=cal_dataset)

# %%
# 6. Iterating Again to Compute the Improved ECE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can directly use the scaler as a calibrated model.
#
# Note that you will need to first reset the ECE metric to avoid mixing the scores of
# the previous and current iterations.

# Reset the ECE
ece.reset()

# Iterate on the test dataloader
for sample, target in test_dataloader:
    logits = scaled_model(sample)
    probs = logits.softmax(-1)
    ece.update(probs, target)

print(f"ECE after scaling - {ece.compute():.3%}.")

# %%
# We finally compute and plot the scaled top-label calibration figure. We see
# that the model is now better calibrated.
fig, ax = ece.plot()
fig.show()

# %%
# The top-label calibration should be improved.
#
# Notes
# ~~~~~
#
# Temperature scaling is very efficient when the calibration set is representative of the test set.
# In this case, we say that the calibration and test set are drawn from the same distribution.
# However, this may not hold true in real-world cases where dataset shift could happen.

# %%
# References
# ~~~~~~~~~~
#
# - **Expected Calibration Error:** Naeini, M. P., Cooper, G. F., & Hauskrecht, M. (2015). Obtaining Well Calibrated Probabilities Using Bayesian Binning. In `AAAI 2015 <https://arxiv.org/pdf/1411.0160.pdf>`_.
# - **Temperature Scaling:** Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On calibration of modern neural networks. In `ICML 2017 <https://arxiv.org/pdf/1706.04599.pdf>`_.
