# ruff: noqa: E402, E703, D212, D415
"""
Training a LeNet for Image Classification with TorchUncertainty
===============================================================

In this tutorial, we will train a LeNet classifier on the MNIST dataset using TorchUncertainty.


1. Loading the utilities
~~~~~~~~~~~~~~~~~~~~~~~~

First, we have to load the following utilities from TorchUncertainty:

- the TUTrainer from TorchUncertainty utils
- the datamodule handling dataloaders: MNISTDataModule from torch_uncertainty.datamodules
- the model: lenet from torch_uncertainty.models
- the classification training & evaluation routine in the torch_uncertainty.routines
- an optimization recipe in the torch_uncertainty.optim_recipes module.

We also need import the neural network utils within `torch.nn`.
"""

# %%
from pathlib import Path

from torch import nn

from torch_uncertainty import TUTrainer
from torch_uncertainty.datamodules import MNISTDataModule
from torch_uncertainty.models.lenet import lenet
from torch_uncertainty.optim_recipes import optim_cifar10_resnet18
from torch_uncertainty.routines import ClassificationRoutine

# %%
# 2. Defining the Model and the Trainer
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the following, we first create the trainer and instantiate
# the datamodule that handles the MNIST dataset,
# dataloaders and transforms. We create the model using the
# blueprint from torch_uncertainty.models.

trainer = TUTrainer(accelerator="gpu", devices=1, max_epochs=2, enable_progress_bar=False)

# datamodule
root = Path("data")
datamodule = MNISTDataModule(root=root, batch_size=128)


model = lenet(
    in_channels=datamodule.num_channels,
    num_classes=datamodule.num_classes,
    dropout_rate=0.4,
)

# %%
# 3. The Loss and the Training Routine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# This is a classification problem, and we use CrossEntropyLoss as the (negative-log-)likelihood.
# We define the training routine using the classification training routine from
# torch_uncertainty.routines. We provide the number of classes
# the optimization recipe and tell the routine that our model is an ensemble at evaluation time.

routine = ClassificationRoutine(
    num_classes=datamodule.num_classes,
    model=model,
    loss=nn.CrossEntropyLoss(),
    optim_recipe=optim_cifar10_resnet18(model),
    is_ensemble=True,
)

# %%
# 4. Gathering Everything and Training the Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# We can now train the model using the trainer. We pass the routine and the datamodule
# to the fit and test methods of the trainer. It will automatically evaluate some uncertainty
# metrics that you will find in the table below.

trainer.fit(model=routine, datamodule=datamodule)
results = trainer.test(model=routine, datamodule=datamodule)

# %%
# 5. Evaluating the Model
# ~~~~~~~~~~~~~~~~~~~~~~~
# Now that the model is trained, let's test it on MNIST.

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision


def imshow(img) -> None:
    npimg = img.numpy()
    npimg = npimg * 0.3081 + 0.1307  # unnormalize
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    plt.tight_layout()
    plt.show()


dataiter = iter(datamodule.val_dataloader())
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images[:6, ...], padding=0))
print("Ground truth labels: ", " ".join(f"{labels[j]}" for j in range(6)))

routine.eval()
logits = routine(images)

probs = torch.nn.functional.softmax(logits, dim=-1)

values, predicted = torch.max(probs, 1)
print(
    "LeNet predictions for the first 6 images: ",
    " ".join([str(image_id.item()) for image_id in predicted[:6]]),
)
