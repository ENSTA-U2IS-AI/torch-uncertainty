Quickstart
==========

.. role:: bash(code)
    :language: bash

Torch Uncertainty comes with different usage levels ranging from specific
PyTorch layers to ready to train lightning-based models. The following
presents a short introduction to each one of them. Let's start with the
highest-level usage.

Using the lightning-based CLI tool
----------------------------------

Procedure
^^^^^^^^^

The library provides a full-fledged trainer which can be used directly, via
CLI. To do so, create a file in the experiments folder and use the `cli_main`
routine, which takes as arguments:

* a Lightning Module corresponding to the model, its own arguments, and
  forward/validation/test logic. For instance, you might use already available
  modules, such as the Packed-Ensembles-style ResNet available at
  torch_uncertainty/baselines/packed/resnet.py
* a Lightning DataModule corresponding to the training, validation, and test
  sets with again its arguments and logic. CIFAR-10/100, ImageNet, and
  ImageNet-200 are available, for instance
* a PyTorch loss such as the torch.nn.CrossEntropyLoss
* a dictionary containing the optimization procedure, namely a scheduler and
  an optimizer. Many procedures are available at torch_uncertainty/optimization_procedures.py
* the path to the data and logs folder, in the example below, the root of the library
* and finally, the name of your model (used for logs)

Move to the directory containing your file and execute the code with :bash:`python3 experiment.py`.
Add lightning arguments such as :bash:`--accelerator gpu --devices "0, 1"` for multi-gpu training, etc.

Example
^^^^^^^

The following code - available in the experiments folder - trains a Packed-Ensembles ResNet on CIFAR10:

.. code:: python

    from pathlib import Path

    import torch.nn as nn

    from torch_uncertainty import cli_main
    from torch_uncertainty.baselines.packed import PackedResNet
    from torch_uncertainty.datamodules import CIFAR10DataModule
    from torch_uncertainty.optimization_procedures import optim_cifar10_resnet18

    root = Path(__file__).parent.absolute().parents[1]
    cli_main(
        PackedResNet,
        CIFAR10DataModule,
        nn.CrossEntropyLoss,
        optim_cifar10_resnet18,
        root,
        "packed",
    )

You may replace the architecture (which should be a Lightning Module), the
Datamodule (a Lightning Datamodule), the loss or the optimization procedure to your likings.

Using the pytorch-based models
------------------------------

Procedure
^^^^^^^^^

If you prefer classic PyTorch pipelines, we provide PyTorch Modules that do not
require Lightning.

1. Check the API reference under the *Models* section to ensure the architecture of your choice is supported by the library.
2. Create a ``torch.nn.Module`` in your training/testing script using one of the provided building functions listed in the API reference.

Example
^^^^^^^

You can initialize a Packed-Ensemble out of a ResNet18
backbone using the following:

.. code:: python

    from torch_uncertainty.models.resnet import packed_resnet18

    model = packed_resnet18(
        in_channels = 3
        num_estimators = 4,
        alpha = 2,
        gamma = 2
        num_classes = 10
    )

Using the pytorch-based layers
------------------------------

Procedure
^^^^^^^^^

It is likely that your desired architecture is not supported by our library.
In that case, you might be interested in directly using the actual layers.

1. Check the API reference for specific layers of your choosing.
2. Import the layers and use them as you would for any vanilla PyTorch layers.

.. tip::

  Do not hesitate to go to the API reference to get better explanations on the
  layer usage.

Example
^^^^^^^

You can create a Packed-Ensemble ``torch.nn.Module`` model with the following:

.. code:: python

  from einops import rearrange
  from torch_uncertainty.layers import PackedConv2d, PackedLinear

  class PackedNet(nn.Module):
      def __init__(self) -> None:
          super().__init__()
          M = 4
          alpha = 2
          gamma = 1
          self.conv1 = PackedConv2d(3, 6, 5, alpha=alpha, num_estimators=M, gamma=gamma, first=True)
          self.pool = nn.MaxPool2d(2, 2)
          self.conv2 = PackedConv2d(6, 16, 5, alpha=alpha, num_estimators=M, gamma=gamma)
          self.fc1 = PackedLinear(16 * 5 * 5, 120, alpha=alpha, num_estimators=M, gamma=gamma)
          self.fc2 = PackedLinear(120, 84, alpha=alpha, num_estimators=M, gamma=gamma)
          self.fc3 = PackedLinear(84, 10, alpha=alpha, num_estimators=M, gamma=gamma, last=True)

          self.num_estimators = M

      def forward(self, x):
          x = self.pool(F.relu(self.conv1(x)))
          x = self.pool(F.relu(self.conv2(x)))
          x = rearrange(
              x, "e (m c) h w -> (m e) c h w", m=self.num_estimators
          )
          x = x.flatten(1)
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = self.fc3(x)
          return x

  packed_net = PackedNet()

Other usage
-----------

Feel free to use any classes described in the API reference such as the metrics, datasets, etc.
