# -*- coding: utf-8 -*-
# fmt: off
# flake: noqa
"""
From a Vanilla Classifier to a Packed-Ensemble
==============================================

Let's dive step by step into the process to modify a vanilla classifier into a
packed-ensemble classifier.

Dataset
-------

In this tutorial we will use the CIFAR10 dataset available in the torchvision
package. The CIFAR10 dataset consists of 60000 32x32 colour images in 10
classes, with 6000 images per class. There are 50000 training images and 10000
test images.

Here is an example of what the data looks like:

.. figure:: /_static/img/cifar10.png
   :alt: cifar10

   cifar10

Training a image Packed-Ensemble classifier
-------------------------------------------

Here is the outline of the process:

1. Load and normalizing the CIFAR10 training and test datasets using
   ``torchvision``
2. Define a Packed-Ensemble from a vanilla classifier
3. Define a loss function
4. Train the Packed-Ensemble on the training data
5. Test the Packed-Ensemble on the test data and evaluate its performance
   w.r.t. uncertainty quantification and OOD detection

1. Load and normalize CIFAR10
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
"""
import torch
import torchvision
import torchvision.transforms as transforms

########################################################################
# The output of torchvision datasets are PILImage images of range [0, 1].
# We transform them to Tensors of normalized range [-1, 1].

########################################################################
# .. note::
#     If running on Windows and you get a BrokenPipeError, try setting
#     the num_worker of torch.utils.data.DataLoader() to 0.

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################
# Let us show some of the training images, for fun.

import matplotlib.pyplot as plt

import numpy as np

# functions to show an image


def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


########################################################################
# 2. Define a Packed-Ensemble from a vanilla classifier
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
