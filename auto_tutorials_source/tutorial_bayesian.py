# -*- coding: utf-8 -*-
# fmt: off
# flake: noqa
"""
Train a Bayesian Neural Network in Three Minutes
================================================

In this tutorial, we will train a Bayesian Neural Network (BNN) LeNet classifier on the MNIST dataset.

Foreword on Bayesian Neural Networks
------------------------------------

Bayesian Neural Networks (BNNs) are a class of neural networks that can estimate the uncertainty of their predictions via uncertainty on their weights. This is achieved by considering the weights of the neural network as random variables, and by learning their posterior distribution. This is in contrast to standard neural networks, which only learn a single set of weights, which can be seen as Dirac distributions on the weights.

For more information on Bayesian Neural Networks, we refer the reader to the following resources:

- Weight Uncertainty in Neural Networks [ICML2015](https://arxiv.org/pdf/1505.05424.pdf)
- Hands-on Bayesian Neural Networks - a Tutorial for Deep Learning Users [IEEE Computational Intelligence Magazine](https://arxiv.org/pdf/2007.06823.pdf)

Training a Bayesian LeNet using TorchUncertainty models and PyTorch Lightning
-----------------------------------------------------------------------------

In this part, we train a bayesian LeNe, based on the already implemented method.

1. Loading the utilities
~~~~~~~~~~~~~~~~~~~~~~~~

To train a BNN using TorchUncertainty, we have to load the following utilities from TorchUncertainty:
- the model: bayesian_lenet, which lies in the torch_uncertainty.model module
- the classification training routine in the torch_uncertainty.training.classification module
- the bayesian objective: the ELBOLoss, which lies in the torch_uncertainty.losses file
- the datamodule that handles dataloaders: MNISTDataModule, which lies in the torch_uncertainty.datamodule
- the cli handler: cli_main and argument parser: init_args
"""

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.datamodules import MNISTDataModule
from torch_uncertainty.losses import ELBOLoss
from torch_uncertainty.models.lenet import bayesian_lenet
from torch_uncertainty.routines.classification import ClassificationSingle

########################################################################
# We will also need to define an optimizer using torch.optim as well as the 
# neural network utils withing torch.nn, as well as the partial util to provide
# the modified default arguments for the ELBO loss.

# We also import ArgvContext to avoid using the jupyter arguments as cli
# arguments, and therefore avoid errors.

import torch.nn as nn
import torch.optim as optim

from functools import partial
from pathlib import Path
import os
from cli_test_helpers import ArgvContext

########################################################################
# Creating the Optimizer Wrapper
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We will use the Adam optimizer with the default learning rate of 0.001.

def optim_lenet(model: nn.Module) -> dict:
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    return {"optimizer": optimizer}

########################################################################
# Creating the necessary variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# In the following, we will need to define the root of the datasets and the
# logs, and to fake parse the arguments needed for using the PyTorch Lightning
# Trainer. We also create the datamodule that handles the MNIST dataset,
# dataloaders and transforms. Finally, we also create the model using the
# blueprint from torch_uncertainty.models. 

root = Path(os.path.abspath(""))

with ArgvContext("--max_epochs 5"): #TODO: understand why it doesn't work
    args = init_args(datamodule=MNISTDataModule)
    args.enable_progress_bar = False
    args.verbose = False

net_name = "bayesian-lenet-mnist"

# datamodule
args.root = str(root / "data")
dm = MNISTDataModule(**vars(args))

# model
model = bayesian_lenet(dm.num_channels, dm.num_classes)

########################################################################
# The Loss and the Training Routine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Then, we just have to define the loss to be used during training. To do this,
# we redefine the default parameters from the ELBO loss using the partial
# function from functools. We use the hyperparameters proposed in the blitz
# library. As we are train a classification model, we use the CrossEntropyLoss
# as the likelihood.
# We then define the training routine using the classification training routine
# from torch_uncertainty.training.classification. We provide the model, the ELBO
# loss and the optimizer, as well as all the default arguments.

loss = partial(
    ELBOLoss,
    model=model,
    criterion=nn.CrossEntropyLoss(),
    kl_weight=1 / 50000,
    num_samples=3,
)

baseline = ClassificationSingle(
    model=model,
    num_classes=dm.num_classes,
    in_channels=dm.num_channels,
    loss=loss,
    optimization_procedure=optim_lenet,
    **vars(args),
)

########################################################################
### Gathering Everything and Train the Model
# Now that we have prepared all of this, we just have to gather everything in
# the main function and to train the model using the PyTorch Lightning Trainer.
# Specifically, it needs the baseline, that includes the model as well as the
# training routine, the datamodule, the root for the datasets and the logs, the
# name of the model for the logs and all the training arguments.
# The dataset will be downloaded automatically in the root/data folder, and the
# logs will be saved in the root/logs folder.

cli_main(baseline, dm, root, net_name, args)

########################################################################
# References
# ----------
# **LeNet & MNIST:**
# LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based
# learning applied to document recognition.
# [Proceedings of the IEEE](vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf).
# **Bayesian Neural Networks:**
# Weight Uncertainty in Neural Networks
# [ICML2015](https://arxiv.org/pdf/1505.05424.pdf)
# **The Adam optimizer:**
# Kingma, Diederik P., and Jimmy Ba. "Adam: A method for stochastic optimization."
# [ICLR 2015](https://arxiv.org/pdf/1412.6980.pdf)
# The [Blitz library](https://github.com/piEsposito/blitz-bayesian-deep-learning/tree/master)
# (for the hyperparameters)
