Quickstart
==========

The following code - available in the experiments folder -trains a Packed-Ensembles ResNet on CIFAR10:

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

You may replace the architecture (which should be a Lightning Module), the Datamodule (a Lightning Datamodule), the loss or the optimization procedure to your likings.