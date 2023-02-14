Quickstart
==========

.. role:: bash(code)
    :language: bash

You can use the Torch Uncertainty library at different levels. Let's start with the highest-level usage.

Using the CLI tool
------------------

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

Move to the directory containing your file and execute the code with :bash:`python3 experiment.py`. Add lightning arguments such as :bash:`--accelerator gpu --devices "0, 1"` for multi-gpu training, etc.

Exemple
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

Using your own trainer
----------------------

For now, the lightning trainer is the only training procedure available.
In the meantime, feel free to reuse:

* the layers at torch_uncertainty/layers
* the models at torch_uncertainty/models
* the metrics at torch_uncertainty/metrics
* the datasets at torch_uncertainty/datasets
