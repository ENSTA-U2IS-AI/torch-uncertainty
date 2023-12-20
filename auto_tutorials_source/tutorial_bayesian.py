"""
Train a Bayesian Neural Network in Three Minutes
================================================

In this tutorial, we will train a Bayesian Neural Network (BNN) LeNet classifier on the MNIST dataset.

Foreword on Bayesian Neural Networks
------------------------------------

Bayesian Neural Networks (BNNs) are a class of neural networks that can estimate the uncertainty of their predictions via uncertainty on their weights. This is achieved by considering the weights of the neural network as random variables, and by learning their posterior distribution. This is in contrast to standard neural networks, which only learn a single set of weights, which can be seen as Dirac distributions on the weights.

For more information on Bayesian Neural Networks, we refer the reader to the following resources:

- Weight Uncertainty in Neural Networks `ICML2015 <https://arxiv.org/pdf/1505.05424.pdf>`_
- Hands-on Bayesian Neural Networks - a Tutorial for Deep Learning Users `IEEE Computational Intelligence Magazine <https://arxiv.org/pdf/2007.06823.pdf>`_

Training a Bayesian LeNet using TorchUncertainty models and PyTorch Lightning
-----------------------------------------------------------------------------

In this part, we train a bayesian LeNet, based on the model and routines already implemented in TU.

1. Loading the utilities
~~~~~~~~~~~~~~~~~~~~~~~~

To train a BNN using TorchUncertainty, we have to load the following utilities from TorchUncertainty:

- the cli handler: cli_main and argument parser: init_args
- the model: bayesian_lenet, which lies in the torch_uncertainty.model module
- the classification training routine in the torch_uncertainty.training.classification module
- the bayesian objective: the ELBOLoss, which lies in the torch_uncertainty.losses file
- the datamodule that handles dataloaders: MNISTDataModule, which lies in the torch_uncertainty.datamodule
"""

from torch_uncertainty import cli_main, init_args
from torch_uncertainty.datamodules import MNISTDataModule
from torch_uncertainty.losses import ELBOLoss
from torch_uncertainty.models.lenet import bayesian_lenet
from torch_uncertainty.routines.classification import ClassificationSingle

# %%
# We will also need to define an optimizer using torch.optim as well as the
# neural network utils withing torch.nn, as well as the partial util to provide
# the modified default arguments for the ELBO loss.
#
# We also import ArgvContext to avoid using the jupyter arguments as cli
# arguments, and therefore avoid errors.

import os
from functools import partial
from pathlib import Path

from torch import nn, optim
from cli_test_helpers import ArgvContext

# %%
# 2. Creating the Optimizer Wrapper
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We will use the Adam optimizer with the default learning rate of 0.001.


def optim_lenet(model: nn.Module) -> dict:
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    return {"optimizer": optimizer}


# %%
# 3. Creating the necessary variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the following, we will need to define the root of the datasets and the
# logs, and to fake-parse the arguments needed for using the PyTorch Lightning
# Trainer. We also create the datamodule that handles the MNIST dataset,
# dataloaders and transforms. Finally, we create the model using the
# blueprint from torch_uncertainty.models.

root = Path(os.path.abspath(""))

# We mock the arguments for the trainer
with ArgvContext(
    "file.py",
    "--max_epochs",
    "1",
    "--enable_progress_bar",
    "False",
):
    args = init_args(datamodule=MNISTDataModule)

net_name = "logs/bayesian-lenet-mnist"

# datamodule
args.root = str(root / "data")
dm = MNISTDataModule(**vars(args))

# model
model = bayesian_lenet(dm.num_channels, dm.num_classes)

# %%
# 4. The Loss and the Training Routine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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

# %%
# 5. Gathering Everything and Training the Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now that we have prepared all of this, we just have to gather everything in
# the main function and to train the model using the PyTorch Lightning Trainer.
# Specifically, it needs the baseline, that includes the model as well as the
# training routine, the datamodule, the root for the datasets and the logs, the
# name of the model for the logs and all the training arguments.
# The dataset will be downloaded automatically in the root/data folder, and the
# logs will be saved in the root/logs folder.

results = cli_main(baseline, dm, root, net_name, args)

# %%
# 6. Testing the Model
# ~~~~~~~~~~~~~~~~~~~~
#
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

# %%
# References
# ----------
#
# - **LeNet & MNIST:** LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. `Proceedings of the IEEE <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.
# - **Bayesian Neural Networks:** Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight Uncertainty in Neural Networks. `ICML 2015 <https://arxiv.org/pdf/1505.05424.pdf>`_.
# - **The Adam optimizer:** Kingma, D. P., & Ba, J. (2014). "Adam: A method for stochastic optimization." `ICLR 2015 <https://arxiv.org/pdf/1412.6980.pdf>`_.
# - **The Blitz** `library <https://github.com/piEsposito/blitz-bayesian-deep-learning>`_ (for the hyperparameters).
