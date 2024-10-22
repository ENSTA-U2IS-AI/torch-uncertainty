"""
Deep Evidential Classification on a Toy Example
===============================================

This tutorial aims to provide an introductory overview of Deep Evidential Classification (DEC) using a practical example. We demonstrate an application of DEC by tackling the toy-problem of fitting the MNIST dataset using a Multi-Layer Perceptron (MLP) neural network model. The output of the MLP is modeled as a Dirichlet distribution. The MLP is trained by minimizing the DEC loss function, composed of a Bayesian risk square error loss and a regularization term based on KL Divergence.

DEC represents an evidential approach to quantifying uncertainty in neural network classification models. This method involves introducing prior distributions over the parameters of the Categorical likelihood function. Then, the MLP model estimates the parameters of the evidential distribution.

Training a LeNet with DEC using TorchUncertainty models
-------------------------------------------------------

In this part, we train a neural network, based on the model and routines already implemented in TU.

1. Loading the utilities
~~~~~~~~~~~~~~~~~~~~~~~~

To train a LeNet with the DEC loss function using TorchUncertainty, we have to load the following utilities from TorchUncertainty:

- our wrapper of the Lightning Trainer
- the model: LeNet, which lies in torch_uncertainty.models
- the classification training routine in the torch_uncertainty.routines
- the evidential objective: the DECLoss from torch_uncertainty.losses
- the datamodule that handles dataloaders & transforms: MNISTDataModule from torch_uncertainty.datamodules

We also need to define an optimizer using torch.optim, the neural network utils within torch.nn.
"""
# %%
from pathlib import Path

import torch
from torch import nn, optim

from torch_uncertainty import TUTrainer
from torch_uncertainty.datamodules import MNISTDataModule
from torch_uncertainty.losses import DECLoss
from torch_uncertainty.models.lenet import lenet
from torch_uncertainty.routines import ClassificationRoutine


# %%
# 2. Creating the Optimizer Wrapper
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We follow the official implementation in DEC, use the Adam optimizer
# with the default learning rate of 0.001 and a step scheduler.
def optim_lenet(model: nn.Module) -> dict:
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    return {"optimizer": optimizer, "lr_scheduler": exp_lr_scheduler}


# %%
# 3. Creating the necessary variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the following, we need to define the root of the logs, and to
# We use the same MNIST classification example as that used in the
# original DEC paper. We only train for 3 epochs for the sake of time.
trainer = TUTrainer(accelerator="cpu", max_epochs=3, enable_progress_bar=False)

# datamodule
root = Path() / "data"
datamodule = MNISTDataModule(root=root, batch_size=128)

model = lenet(
    in_channels=datamodule.num_channels,
    num_classes=datamodule.num_classes,
)

# %%
# 4. The Loss and the Training Routine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Next, we need to define the loss to be used during training.
# After that, we define the training routine using
# the single classification model training routine from
# torch_uncertainty.routines.ClassificationRoutine.
# In this routine, we provide the model, the DEC loss, the optimizer,
# and all the default arguments.

loss = DECLoss(reg_weight=1e-2)

routine = ClassificationRoutine(
    model=model,
    num_classes=datamodule.num_classes,
    loss=loss,
    optim_recipe=optim_lenet(model),
)

# %%
# 5. Gathering Everything and Training the Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

trainer.fit(model=routine, datamodule=datamodule)
trainer.test(model=routine, datamodule=datamodule)

# %%
# 6. Testing the Model
# ~~~~~~~~~~~~~~~~~~~~
# Now that the model is trained, let's test it on MNIST.

import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms.functional as F


def imshow(img) -> None:
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def rotated_mnist(angle: int) -> None:
    """Rotate MNIST images and show images and confidence.

    Args:
        angle: Rotation angle in degrees.
    """
    rotated_images = F.rotate(images, angle)
    # print rotated images
    plt.axis("off")
    imshow(torchvision.utils.make_grid(rotated_images[:4, ...]))
    print("Ground truth: ", " ".join(f"{labels[j]}" for j in range(4)))

    evidence = routine(rotated_images)
    alpha = torch.relu(evidence) + 1
    strength = torch.sum(alpha, dim=1, keepdim=True)
    probs = alpha / strength
    entropy = -1 * torch.sum(probs * torch.log(probs), dim=1, keepdim=True)
    for j in range(4):
        predicted = torch.argmax(probs[j, :])
        print(
            f"Predicted digits for the image {j}: {predicted} with strength "
            f"{strength[j,0]:.3} and entropy {entropy[j,0]:.3}."
        )


dataiter = iter(datamodule.val_dataloader())
images, labels = next(dataiter)

with torch.no_grad():
    routine.eval()
    rotated_mnist(0)
    rotated_mnist(45)
    rotated_mnist(90)

# %%
# References
# ----------
#
# - **Deep Evidential Classification:** Murat Sensoy, Lance Kaplan, & Melih Kandemir (2018). Evidential Deep Learning to Quantify Classification Uncertainty `NeurIPS 2018 <https://arxiv.org/pdf/1806.01768>`_.
