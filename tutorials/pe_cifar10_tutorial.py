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
# First let's define a vanilla classifier for CIFAR10. We will use a
# convolutional neural network.

import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
      def __init__(self):
         super(Net, self).__init__()
         self.conv1 = nn.Conv2d(3, 6, 5)
         self.pool = nn.MaxPool2d(2, 2)
         self.conv2 = nn.Conv2d(6, 16, 5)
         self.fc1 = nn.Linear(16 * 5 * 5, 120)
         self.fc2 = nn.Linear(120, 84)
         self.fc3 = nn.Linear(84, 10)
   
      def forward(self, x):
         x = self.pool(F.relu(self.conv1(x)))
         x = self.pool(F.relu(self.conv2(x)))
         x = x.flatten(1)
         x = F.relu(self.fc1(x))
         x = F.relu(self.fc2(x))
         x = self.fc3(x)
         return x

net = Net()

########################################################################
# You can now modify the vanilla classifier into a Packed-Ensemble
# (4,2,1) by using the :func:`packing()` function:

from torch_uncertainty import packing

packing(net, num_estimators=2, alpha=2, gamma=1)

print(net)

########################################################################
# 3. Define a Loss function and optimizer
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's use a Classification Cross-Entropy loss and SGD with momentum.

import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the Packed-Ensemble on the training data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's train the Packed-Ensemble on the training data.

for epoch in range(2):  # loop over the dataset multiple times
    
   running_loss = 0.0
   for i, data in enumerate(trainloader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      if i % 2000 == 1999:    # print every 2000 mini-batches
         print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
         running_loss = 0.0
