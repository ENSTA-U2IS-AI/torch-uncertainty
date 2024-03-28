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

- the Trainer from Lightning
- the datamodule that handles dataloaders: MNISTDataModule, which lies in the torch_uncertainty.datamodule
- the model: LeNet, which lies in torch_uncertainty.models
- the mc-dropout wrapper: mc_dropout, which lies in torch_uncertainty.models
- the classification training routine in the torch_uncertainty.training.classification module
- an optimization recipe in the torch_uncertainty.optim_recipes module.

We also need import the neural network utils within `torch.nn`.
"""
# %%
from pathlib import Path

from lightning.pytorch import Trainer
from torch import nn

from torch_uncertainty.datamodules import MNISTDataModule
from torch_uncertainty.models.lenet import lenet
from torch_uncertainty.models import mc_dropout
from torch_uncertainty.optim_recipes import optim_cifar10_resnet18
from torch_uncertainty.routines import ClassificationRoutine

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

trainer = Trainer(accelerator="cpu", max_epochs=2, enable_progress_bar=False)

# datamodule
root = Path("")  / "data"
datamodule = MNISTDataModule(root=root, batch_size=128)


model = lenet(
    in_channels=datamodule.num_channels,
    num_classes=datamodule.num_classes,
    dropout_rate=0.6,
)

mc_model = mc_dropout(model, num_estimators=16, last_layer=False)

# %%
# 3. The Loss and the Training Routine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is a classification problem, and we use CrossEntropyLoss as the likelihood.
# We define the training routine using the classification training routine from
# torch_uncertainty.routines.classification. We provide the number of classes
# and channels, the optimizer wrapper, the dropout rate, and the number of
# forward passes to perform through the network, as well as all the default
# arguments.

routine = ClassificationRoutine(
    num_classes=datamodule.num_classes,
    model=mc_model,
    loss=nn.CrossEntropyLoss(),
    optim_recipe=optim_cifar10_resnet18(mc_model),
    num_estimators=16,

)

# %%
# 4. Gathering Everything and Training the Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

trainer.fit(model=routine, datamodule=datamodule)
trainer.test(model=routine, datamodule=datamodule)

# %%
# 5. Testing the Model
# ~~~~~~~~~~~~~~~~~~~~
# Now that the model is trained, let's test it on MNIST. Don't forget to call
# .eval() to enable dropout at inference.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


dataiter = iter(datamodule.val_dataloader())
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images[:4, ...]))
print("Ground truth: ", " ".join(f"{labels[j]}" for j in range(4)))

routine.eval()
logits = routine(images).reshape(16, 128, 10)

probs = torch.nn.functional.softmax(logits, dim=-1)


for j in range(4):
    values, predicted = torch.max(probs[:, j], 1)
    print(
        f"Predicted digits for the image {j+1}: ",
        " ".join([str(image_id.item()) for image_id in predicted]),
    )

# %% We see that there is some disagreement between the samples of the dropout
# approximation of the posterior distribution.
