"""
Training a LeNet with Monte-Carlo Dropout
=========================================

In this tutorial, we will train a LeNet classifier on the MNIST dataset using Monte-Carlo Dropout (MC Dropout), a computationally efficient Bayesian approximation method. To estimate the predictive mean and uncertainty (variance), we perform multiple forward passes through the network with dropout layers enabled in ``train`` mode.

For more information on Monte-Carlo Dropout, we refer the reader to the following resources:

- What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision? `NeurIPS 2017 <https://browse.arxiv.org/pdf/1703.04977.pdf>`_
- Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning `PMLR 2016 <https://browse.arxiv.org/pdf/1506.02142.pdf>`_

Training a LeNet with MC Dropout using TorchUncertainty models and PyTorch Lightning
-------------------------------------------------------------------------------------

In this part, we train a LeNet with dropout layers, based on the model and routines already implemented in TU.

1. Loading the utilities
~~~~~~~~~~~~~~~~~~~~~~~~

First, we have to load the following utilities from TorchUncertainty:

- the cli handler: cli_main and argument parser: init_args
- the datamodule that handles dataloaders: MNISTDataModule, which lies in the torch_uncertainty.datamodule
- the model: LeNet, which lies in torch_uncertainty.models
- the mc-dropout wrapper: mc_dropout, which lies in torch_uncertainty.models
- a resnet baseline to get the command line arguments: ResNet, which lies in torch_uncertainty.baselines
- the classification training routine in the torch_uncertainty.training.classification module
- the optimizer wrapper in the torch_uncertainty.optimization_procedures module.
"""
# %%
from torch_uncertainty import cli_main, init_args
from torch_uncertainty.datamodules import MNISTDataModule
from torch_uncertainty.models.lenet import lenet
from torch_uncertainty.models.mc_dropout import mc_dropout
from torch_uncertainty.baselines.classification import ResNet
from torch_uncertainty.routines.classification import ClassificationEnsemble
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
    "--dropout_rate",
    "0.6",
    "--num_estimators",
    "16",
    "--max_epochs",
    "2"
):
    args = init_args(network=ResNet, datamodule=MNISTDataModule)

net_name = "logs/mc-dropout-lenet-mnist"

# datamodule
args.root = str(root / "data")
dm = MNISTDataModule(**vars(args))


model = lenet(
    in_channels=dm.num_channels,
    num_classes=dm.num_classes,
    dropout_rate=args.dropout_rate,
)

mc_model = mc_dropout(model, num_estimators=args.num_estimators, last_layer=0.0)

# %%
# 3. The Loss and the Training Routine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is a classification problem, and we use CrossEntropyLoss as the likelihood.
# We define the training routine using the classification training routine from
# torch_uncertainty.training.classification. We provide the number of classes
# and channels, the optimizer wrapper, the dropout rate, and the number of
# forward passes to perform through the network, as well as all the default
# arguments.

baseline = ClassificationEnsemble(
    num_classes=dm.num_classes,
    model=mc_model,
    loss=nn.CrossEntropyLoss,
    optimization_procedure=optim_cifar10_resnet18,
    **vars(args),
)

# %%
# 5. Gathering Everything and Training the Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

results = cli_main(baseline, dm, root, net_name, args)

# %%
# 6. Testing the Model
# ~~~~~~~~~~~~~~~~~~~~
# Now that the model is trained, let's test it on MNIST. Don't forget to call
# .eval() to enable dropout at inference.

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
logits = baseline(images).reshape(16, 128, 10)

probs = torch.nn.functional.softmax(logits, dim=-1)


for j in range(4):
    values, predicted = torch.max(probs[:, j], 1)
    print(
        f"Predicted digits for the image {j}: ",
        " ".join([str(image_id.item()) for image_id in predicted]),
    )

# %% We see that there is some disagreement between the samples of the dropout
# approximation of the posterior distribution.
