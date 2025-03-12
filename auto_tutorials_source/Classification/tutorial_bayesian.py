"""
Train a Bayesian Neural Network in Three Minutes
================================================

In this tutorial, we will train a variational inference Bayesian Neural Network (BNN) LeNet classifier on the MNIST dataset.

Foreword on Bayesian Neural Networks
------------------------------------

Bayesian Neural Networks (BNNs) are a class of neural networks that estimate the uncertainty on their predictions via uncertainty
on their weights. This is achieved by considering the weights of the neural network as random variables, and by learning their
posterior distribution. This is in contrast to standard neural networks, which only learn a single set of weights, which can be
seen as Dirac distributions on the weights.

For more information on Bayesian Neural Networks, we refer the reader to the following resources:

- Weight Uncertainty in Neural Networks `ICML2015 <https://arxiv.org/pdf/1505.05424.pdf>`_
- Hands-on Bayesian Neural Networks - a Tutorial for Deep Learning Users `IEEE Computational Intelligence Magazine <https://arxiv.org/pdf/2007.06823.pdf>`_

Training a Bayesian LeNet using TorchUncertainty models and Lightning
---------------------------------------------------------------------

In this part, we train a Bayesian LeNet, based on the model and routines already implemented in TU.

1. Loading the utilities
~~~~~~~~~~~~~~~~~~~~~~~~

To train a BNN using TorchUncertainty, we have to load the following modules:

- our TUTrainer
- the model: bayesian_lenet, which lies in the torch_uncertainty.model
- the classification training routine from torch_uncertainty.routines
- the Bayesian objective: the ELBOLoss, which lies in the torch_uncertainty.losses file
- the datamodule that handles dataloaders: MNISTDataModule from torch_uncertainty.datamodules

We will also need to define an optimizer using torch.optim and Pytorch's
neural network utils from torch.nn.
"""
# %%
from pathlib import Path

from torch import nn, optim

from torch_uncertainty import TUTrainer
from torch_uncertainty.datamodules import MNISTDataModule
from torch_uncertainty.losses import ELBOLoss
from torch_uncertainty.models.lenet import bayesian_lenet
from torch_uncertainty.routines import ClassificationRoutine

# %%
# 2. The Optimization Recipe
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# We will use the Adam optimizer with the default learning rate of 0.001.


def optim_lenet(model: nn.Module):
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-3,
    )
    return optimizer


# %%
# 3. Creating the necessary variables
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# In the following, we instantiate our trainer, define the root of the datasets and the logs.
# We also create the datamodule that handles the MNIST dataset, dataloaders and transforms.
# Please note that the datamodules can also handle OOD detection by setting the eval_ood
# parameter to True. Finally, we create the model using the blueprint from torch_uncertainty.models.

trainer = TUTrainer(accelerator="cpu", enable_progress_bar=False, max_epochs=1)

# datamodule
root = Path("data")
datamodule = MNISTDataModule(root=root, batch_size=128, eval_ood=False)

# model
model = bayesian_lenet(datamodule.num_channels, datamodule.num_classes)

# %%
# 4. The Loss and the Training Routine
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Then, we just have to define the loss to be used during training. To do this,
# we redefine the default parameters from the ELBO loss using the partial
# function from functools. We use the hyperparameters proposed in the blitz
# library. As we are train a classification model, we use the CrossEntropyLoss
# as the likelihood.
# We then define the training routine using the classification training routine
# from torch_uncertainty.classification. We provide the model, the ELBO
# loss and the optimizer to the routine.

loss = ELBOLoss(
    model=model,
    inner_loss=nn.CrossEntropyLoss(),
    kl_weight=1 / 10000,
    num_samples=3,
)

routine = ClassificationRoutine(
    model=model,
    num_classes=datamodule.num_classes,
    loss=loss,
    optim_recipe=optim_lenet(model),
    is_ensemble=True
)

# %%
# 5. Gathering Everything and Training the Model
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Now that we have prepared all of this, we just have to gather everything in
# the main function and to train the model using our wrapper of Lightning Trainer.
# Specifically, it needs the routine, that includes the model as well as the
# training/eval logic and the datamodule
# The dataset will be downloaded automatically in the root/data folder, and the
# logs will be saved in the root/logs folder.

trainer.fit(model=routine, datamodule=datamodule)
trainer.test(model=routine, datamodule=datamodule)

# %%
# 6. Testing the Model
# ~~~~~~~~~~~~~~~~~~~~
#
# Now that the model is trained, let's test it on MNIST.
# Please note that we apply a reshape to the logits to determine the dimension corresponding to the ensemble
# and to the batch. As for TorchUncertainty 0.2.0, the ensemble dimension is merged with the batch dimension
# in this order (num_estimator x batch, classes).
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

# Put the model in eval mode to use several samples
model = model.eval()
logits = model(images).reshape(16, 128, 10) # num_estimators, batch_size, num_classes

# We apply the softmax on the classes and average over the estimators
probs = torch.nn.functional.softmax(logits, dim=-1)
avg_probs = probs.mean(dim=0)
var_probs = probs.std(dim=0)

_, predicted = torch.max(avg_probs, 1)

print("Predicted digits: ", " ".join(f"{predicted[j]}" for j in range(4)))
print("Std. dev. of the scores over the posterior samples", " ".join(f"{var_probs[j][predicted[j]]:.3}" for j in range(4)))
# %%
# Here, we show the variance of the top prediction. This is a non-standard but intuitive way to show the diversity of the predictions
# of the ensemble. Ideally, the variance should be high when the average top prediction is incorrect.
#
# References
# ----------
#
# - **LeNet & MNIST:** LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. `Proceedings of the IEEE <http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf>`_.
# - **Bayesian Neural Networks:** Blundell, C., Cornebise, J., Kavukcuoglu, K., & Wierstra, D. (2015). Weight Uncertainty in Neural Networks. `ICML 2015 <https://arxiv.org/pdf/1505.05424.pdf>`_.
# - **The Adam optimizer:** Kingma, D. P., & Ba, J. (2014). "Adam: A method for stochastic optimization." `ICLR 2015 <https://arxiv.org/pdf/1412.6980.pdf>`_.
# - **The Blitz** `library <https://github.com/piEsposito/blitz-bayesian-deep-learning>`_ (for the hyperparameters).
