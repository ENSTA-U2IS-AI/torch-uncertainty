CLI Guide
=========

Introduction to the Lightning CLI
---------------------------------

The Lightning CLI tool eases the implementation of a CLI to instanciate models to train and evaluate them on
some data. The CLI tool is a wrapper around the ``Trainer`` class and provides a set of subcommands to train
and test a ``LightningModule`` on a ``LightningDataModule``. To better match our needs, we created an inherited
class from the ``LightningCLI`` class, namely ``TULightningCLI``.

.. note::
    ``TULightningCLI`` adds a new argument to the ``LightningCLI`` class: :attr:`eval_after_fit` to know whether
    an evaluation on the test set should be performed after the training phase.

Let's see how to implement the CLI, by checking out the ``experiments/classification/cifar10/resnet.py``.

.. code:: python

    import torch
    from lightning.pytorch.cli import LightningArgumentParser

    from torch_uncertainty.baselines.classification import ResNetBaseline
    from torch_uncertainty.datamodules import CIFAR10DataModule
    from torch_uncertainty import TULightningCLI


    class ResNetCLI(TULightningCLI):
        def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
            parser.add_optimizer_args(torch.optim.SGD)
            parser.add_lr_scheduler_args(torch.optim.lr_scheduler.MultiStepLR)


    def cli_main() -> ResNetCLI:
        return ResNetCLI(ResNetBaseline, CIFAR10DataModule)


    if __name__ == "__main__":
        cli = cli_main()
        if (
            (not cli.trainer.fast_dev_run)
            and cli.subcommand == "fit"
            and cli._get(cli.config, "eval_after_fit")
        ):
            cli.trainer.test(datamodule=cli.datamodule, ckpt_path="best")

This file enables both training and testing ResNet architectures on the CIFAR-10 dataset.
The ``ResNetCLI`` class inherits from the ``TULightningCLI`` class and implements the
``add_arguments_to_parser`` method to add the optimizer and learning rate scheduler arguments
into the parser. In this case, we use the ``torch.optim.SGD`` optimizer and the
``torch.optim.lr_scheduler.MultiStepLR`` learning rate scheduler.

.. code:: python

    class ResNetCLI(TULightningCLI):
        def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
            parser.add_optimizer_args(torch.optim.SGD)
            parser.add_lr_scheduler_args(torch.optim.lr_scheduler.MultiStepLR)

The ``LightningCLI`` takes a ``LightningModule`` and a ``LightningDataModule`` as arguments.
Here the ``cli_main`` function creates an instance of the ``ResNetCLI`` class by taking the ``ResNetBaseline``
model and the ``CIFAR10DataModule`` as arguments.

.. code:: python

    def cli_main() -> ResNetCLI:
        return ResNetCLI(ResNetBaseline, CIFAR10DataModule)

.. note::

    The ``ResNetBaseline`` is a subclass of the ``ClassificationRoutine`` seemlessly instanciating a
    ResNet model based on a :attr:`version` and an :attr:`arch` to be passed to the routine.

Depending on the CLI subcommand calling ``cli_main()`` will either train or test the model on the using
the CIFAR-10 dataset. But what are these subcommands?

.. parsed-literal::

    python resnet.py --help

This command will display the available subcommands of the CLI tool.

.. parsed-literal::

    subcommands:
    For more details of each subcommand, add it as an argument followed by --help.

    Available subcommands:
        fit                 Runs the full optimization routine.
        validate            Perform one evaluation epoch over the validation set.
        test                Perform one evaluation epoch over the test set.
        predict             Run evaluation on your data.

You can execute whichever subcommand you like and set up all your hyperparameters directly using the command line

.. parsed-literal::

    python resnet.py fit --trainer.max_epochs 75 --trainer.accelerators gpu --trainer.devices 1 --model.version std --model.arch 18 --model.in_channels 3 --model.num_classes 10 --model.loss CrossEntropyLoss --model.style cifar --data.root ./data --data.batch_size 128 --optimizer.lr 0.05 --lr_scheduler.milestones [25,50]

All arguments in the ``__init__()`` methods of the ``Trainer``, ``LightningModule`` (here ``ResNetBaseline``),
``LightningDataModule`` (here ``CIFAR10DataModule``), ``torch.optim.SGD``, and ``torch.optim.lr_scheduler.MultiStepLR``
classes are configurable using the CLI tool using the ``--trainer``, ``--model``, ``--data``, ``--optimizer``, and
``--lr_scheduler`` prefixes, respectively.

However for a large number of hyperparameters, it is not practical to pass them all in the command line.
It is more convenient to use configuration files to store these hyperparameters and ease the burden of
repeating them each time you want to train or test a model. Let's see how to do that.

.. note::

    Note that ``Pytorch`` classes are supported by the CLI tool, so you can use them directly: ``--model.loss CrossEntropyLoss``
    and they would be automatically instanciated by the CLI tool with their default arguments (i.e., ``CrossEntropyLoss()``).

.. tip::

    Add the following after calling ``cli=cli_main()`` to eventually evaluate the model on the test set
    after training, if the ``eval_after_fit`` argument is set to ``True`` and ``trainer.fast_dev_run``
    is set to ``False``.

    .. code:: python

        if (
            (not cli.trainer.fast_dev_run)
            and cli.subcommand == "fit"
            and cli._get(cli.config, "eval_after_fit")
        ):
            cli.trainer.test(datamodule=cli.datamodule, ckpt_path="best")

Configuration files
-------------------

By default the ``LightningCLI`` support configuration files in the YAML format (learn more about this format
`here <https://lightning.ai/docs/pytorch/stable/cli/lightning_cli_faq.html#what-is-a-yaml-config-file>`_).
Taking the previous example, we can create a configuration file named ``config.yaml`` with the following content:

.. code:: yaml

    # config.yaml
    eval_after_fit: true
    trainer:
      max_epochs: 75
      accelerators: gpu
      devices: 1
    model:
      version: std
      arch: 18
      in_channels: 3
      num_classes: 10
      loss: CrossEntropyLoss
      style: cifar
    data:
      root: ./data
      batch_size: 128
    optimizer:
      lr: 0.05
    lr_scheduler:
      milestones:
        - 25
        - 50

Then, we can run the following command to train the model:

.. parsed-literal::

    python resnet.py fit --config config.yaml

By default, executing the command above will store the experiment results in a directory named ``lightning_logs``,
and the last state of the model will be saved in a directory named ``lightning_logs/version_{int}/checkpoints``.
In addition, all arguments passed to instanciate the ``Trainer``, ``ResNetBaseline``, ``CIFAR10DataModule``,
``torch.optim.SGD``, and ``torch.optim.lr_scheduler.MultiStepLR`` classes will be saved in a file named
``lightning_logs/version_{int}/config.yaml``. When testing the model, we advise to use this configuration file
to ensure that the same hyperparameters are used for training and testing.

.. parsed-literal::

    python resnet.py test --config lightning_logs/version_{int}/config.yaml --ckpt_path lightning_logs/version_{int}/checkpoints/{filename}.ckpt

Experiment folder usage
-----------------------

Now that we have seen how to implement the CLI tool and how to use configuration files, let explore the
configurations available in the ``experiments`` directory. The ``experiments`` directory is
mainly organized as follows:

.. code:: bash

    experiments
    ├── classification
    │   ├── cifar10
    │   │   ├── configs
    │   │   ├── resnet.py
    │   │   ├── vgg.py
    │   │   └── wideresnet.py
    │   └── cifar100
    │       ├── configs
    │       ├── resnet.py
    │       ├── vgg.py
    │       └── wideresnet.py
    ├── regression
    │   └── uci_datasets
    │       ├── configs
    │       └── mlp.py
    └── segmentation
        ├── cityscapes
        │   ├── configs
        │   └── segformer.py
        └── muad
            ├── configs
            └── segformer.py

For each task (**classification**, **regression**, and **segmentation**), we have a directory containing the datasets
(e.g., CIFAR10, CIFAR100, UCI datasets, Cityscapes, and Muad) and for each dataset, we have a directory containing
the configuration files and the CLI files for different backbones.

You can directly use the CLI files with the command line or use the predefined configuration files to train and test
the models. The configuration files are stored in the ``configs``. For example, the configuration file for the classic
ResNet-18 model on the CIFAR-10 dataset is stored in the ``experiments/classification/cifar10/configs/resnet18/standard.yaml``
file. For the Packed ResNet-18 model on the CIFAR-10 dataset, the configuration file is stored in the
``experiments/classification/cifar10/configs/resnet18/packed.yaml`` file.

If you are interested in using a ResNet model but want to choose some of the hyperparameters using the command line,
you can use the configuration file and override the hyperparameters using the command line. For example, to train
a ResNet-18 model on the CIFAR-10 dataset with a batch size of :math:`256`, you can use the following command:

.. parsed-literal::

    python resnet.py fit --config configs/resnet18/standard.yaml --data.batch_size 256

To use the weights argument of the ``torch.nn.CrossEntropyLoss`` class, you can use the following command:

.. parsed-literal::

    python resnet.py fit --config configs/resnet18/standard.yaml --model.loss CrossEntropyLoss --model.loss.weight Tensor --model.loss.weight.dict_kwargs.data [1,2,3,4,5,6,7,8,9,10]


In addition, we provide a default configuration file for some backbones in the ``configs`` directory. For example,
``experiments/classification/cifar10/configs/resnet.yaml`` contains the default hyperparameters to train a ResNet model
on the CIFAR-10 dataset. Yet, some hyperparameters are purposely missing to be set by the user using the command line.

For instance, to train a Packed ResNet-34 model on the CIFAR-10 dataset with :math:`4` estimators and a :math:`\alpha` value of :math:`2`,
you can use the following command:

.. parsed-literal::

    python resnet.py fit --config configs/resnet.yaml --trainer.max_epochs 75 --model.version packed --model.arch 34 --model.num_estimators 4 --model.alpha 2 --optimizer.lr 0.05 --lr_scheduler.milestones [25,50]


.. tip::

    Explore the `Lightning CLI docs <https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html>`_ to learn more about the CLI tool,
    the available arguments, and how to use them with configuration files.
