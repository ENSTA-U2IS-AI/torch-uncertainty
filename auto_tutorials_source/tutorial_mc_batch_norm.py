"""
Training a LeNet with Monte Carlo Batch Normalization
=====================================================

In this tutorial, we will train a LeNet classifier on the MNIST dataset using Monte-Carlo Batch Normalization (MCBN), a post-hoc Bayesian approximation method. 

Training a LeNet with MCBN using TorchUncertainty models and PyTorch Lightning
------------------------------------------------------------------------------
In this part, we train a LeNet with batch normalization layers, based on the model and routines already implemented in TU.

1. Loading the utilities
~~~~~~~~~~~~~~~~~~~~~~~~

First, we have to load the following utilities from TorchUncertainty:

- the cli handler: cli_main and argument parser: init_args
- the datamodule that handles dataloaders: MNISTDataModule, which lies in the torch_uncertainty.datamodule
- the model: LeNet, which lies in torch_uncertainty.models
- the mc-batch-norm wrapper: mc_dropout, which lies in torch_uncertainty.models
- a resnet baseline to get the command line arguments: ResNet, which lies in torch_uncertainty.baselines
- the classification training routine in the torch_uncertainty.training.classification module
- the optimizer wrapper in the torch_uncertainty.optimization_procedures module.
"""
# %%
from torch_uncertainty import cli_main, init_args
from torch_uncertainty.datamodules import MNISTDataModule
from torch_uncertainty.models.lenet import lenet
from torch_uncertainty.post_processing.mc_batch_norm import MCBatchNorm
from torch_uncertainty.baselines.classification import ResNet
from torch_uncertainty.routines.classification import ClassificationSingle
from torch_uncertainty.optimization_procedures import optim_cifar10_resnet18

# %%
# We will also need import the neural network utils withing `torch.nn`.
#
# We also import ArgvContext to avoid using the jupyter arguments as cli
# arguments, and therefore avoid errors.

import os
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
# dataloaders and transforms. We create the model using the
# blueprint from torch_uncertainty.models and we wrap it into mc-dropout.
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
    "--num_estimators",
    "8",
    "--max_epochs",
    "2"
):
    args = init_args(network=ResNet, datamodule=MNISTDataModule)

net_name = "logs/lenet-mnist"

# datamodule
args.root = str(root / "data")
dm = MNISTDataModule(**vars(args))


model = lenet(
    in_channels=dm.num_channels,
    num_classes=dm.num_classes,
    norm = nn.BatchNorm2d,
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

baseline = ClassificationSingle(
    num_classes=dm.num_classes,
    model=model,
    loss=nn.CrossEntropyLoss,
    optimization_procedure=optim_cifar10_resnet18,
    **vars(args),
)

# %%
# 5. Gathering Everything and Training the Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

results = cli_main(baseline, dm, root, net_name, args)


# %%
# 6. Wrapping the Model in a MCBatchNorm
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can now wrap the model in a MCBatchNorm to add stochasticity to the
# predictions. We specify that the BatchNorm layers are to be converted to
# MCBatchNorm layers, and that we want to use 8 stochastic estimators.
# The amount of stochasticity is controlled by the ``mc_batch_size`` argument.
# The larger the ``mc_batch_size``, the more stochastic the predictions will be.
# The authors suggest 32 as a good value for ``mc_batch_size`` but we use 4 here
# to highlight the effect of stochasticity on the predictions.

baseline.model = MCBatchNorm(baseline.model, num_estimators=8, convert=True, mc_batch_size=32)
baseline.model.fit(dm.train)
baseline.eval()

# %%
# 7. Testing the Model
# ~~~~~~~~~~~~~~~~~~~~
# Now that the model is trained, let's test it on MNIST. Don't forget to call
# .eval() to enable Monte Carlo batch normalization at inference.
# In this tutorial, we plot the most uncertain images, i.e. the images for which
# the variance of the predictions is the highest.

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

baseline.eval()
logits = baseline(images).reshape(8, 128, 10)

probs = torch.nn.functional.softmax(logits, dim=-1)


for j in sorted(probs.var(0).sum(-1).topk(4).indices):
    values, predicted = torch.max(probs[:, j], 1)
    print(
        f"Predicted digits for the image {j}: ",
        " ".join([str(image_id.item()) for image_id in predicted]),
    )

# %%
# The predictions are mostly erroneous, which is expected since we selected
# the most uncertain images. We also see that there stochasticity in the
# predictions, as the predictions for the same image differ depending on the
# stochastic estimator used.
