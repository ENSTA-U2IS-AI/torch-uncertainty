Quickstart
==========

The following code train a Packed-Ensembles ResNet on CIFAR10:

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