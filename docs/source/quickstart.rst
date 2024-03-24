Quickstart
==========

.. role:: bash(code)
    :language: bash

TorchUncertainty is centered around **uncertainty-aware** training and evaluation routines.
These routines make it very easy to:

- train ensembles-like methods (Deep Ensembles, Packed-Ensembles, MIMO, Masksembles, etc)
- compute and monitor uncertainty metrics: calibration, out-of-distribution detection, proper scores, grouping loss, etc.
- leverage calibration methods automatically during evaluation

Yet, we take account that their will be as many different uses of TorchUncertainty as there are of users. 
This page provides ideas on how to benefit from TorchUncertainty at all levels: from ready-to-train lightning-based models to using only specific
PyTorch layers. 

Training with TorchUncertainty's Uncertainty-aware Routines
-----------------------------------------------------------

Let's have a look at the `Classification routine <https://github.com/ENSTA-U2IS-AI/torch-uncertainty/blob/main/torch_uncertainty/routines/classification.py>`_. 

.. code:: python
  from lightning.pytorch import LightningModule

  class ClassificationRoutine(LightningModule):
    def __init__(
      self,
      model: nn.Module,
      num_classes: int,
      loss: nn.Module,
      num_estimators: int = 1,
      format_batch_fn: nn.Module | None = None,
      optim_recipe: dict | Optimizer | None = None,
      mixtype: str = "erm",
      mixmode: str = "elem",
      dist_sim: str = "emb",
      kernel_tau_max: float = 1.0,
      kernel_tau_std: float = 0.5,
      mixup_alpha: float = 0,
      cutmix_alpha: float = 0,
      eval_ood: bool = False,
      eval_grouping_loss: bool = False,
      ood_criterion: Literal[
          "msp", "logit", "energy", "entropy", "mi", "vr"
      ] = "msp",
      log_plots: bool = False,
      save_in_csv: bool = False,
      calibration_set: Literal["val", "test"] | None = None,
    ) -> None:


Building your First Routine
^^^^^^^^^^^^^^^^^^^^^^^^^^^
This routine is a wrapper of any custom or TorchUncertainty classification model. To use it, 
just build your model and pass it to the routine as argument along with the optimization criterion (the loss)
as well as the number of classes that we use for torch metrics. 

.. code:: python
  model = MyModel(num_classes=10)
  routine = ClassificationRoutine(model, num_classes=10, loss=nn.CrossEntropyLoss())


Training with the Routine
^^^^^^^^^^^^^^^^^^^^^^^^^

To train with this routine, you will first need to create a lightning Trainer and have either a lightning datamodule
or PyTorch dataloaders. When benchmarking models, we advise to use lightning datamodules that will automatically handle
train/val/test splits, out-of-distribution detection and dataset shift. For this example, let us use TorchUncertainty's 
CIFAR10 datamodule. Please keep in mind that you could use your own datamodule or dataloaders.

.. code:: python
  from torch_uncertainty.datamodules import CIFAR10DataModule
  from pytorch_lightning import Trainer

  dm = CIFAR10DataModule(root="data", batch_size=32)
  trainer = Trainer(gpus=1, max_epochs=100)
  trainer.fit(routine, dm)
  trainer.eval(routine, dm)

Here it is, you have trained your first model with TorchUncertainty! As a result, you will get access to various metrics
measuring the ability of your model to handle uncertainty.

More metrics
^^^^^^^^^^^^

With TorchUncertainty datamodules, you can easily test models on out-of-distribution datasets, by
setting the `eval_ood` parameter to True. You can also evaluate the grouping loss by setting `eval_grouping_loss` to True.
Finally, you can calibrate your model using the `calibration_set` parameter. In this case, you will get 
metrics for but the uncalibrated and calibrated models: the metrics corresponding to the temperature scaled
model will begin with `ts_`.

Using the Lightning-based CLI tool
----------------------------------

Procedure
^^^^^^^^^

The library provides a full-fledged trainer which can be used directly, via
CLI. To do so, create a file in the experiments folder and use the `cli_main`
routine, which takes as arguments:

* a Lightning Module corresponding to the model, its own arguments, and
  forward/validation/test logic. For instance, you might use already available
  modules, such as the Packed-Ensembles-style ResNet available at
  `torch_uncertainty/baselines/packed/resnet.py <https://github.com/ENSTA-U2IS-AI/torch-uncertainty/blob/main/torch_uncertainty/baselines/classification/resnet.py>`_
* a Lightning DataModule corresponding to the training, validation, and test
  sets with again its arguments and logic. CIFAR-10/100, ImageNet, and
  ImageNet-200 are available, for instance.
* a PyTorch loss such as the torch.nn.CrossEntropyLoss
* a dictionary containing the optimization recipe, namely a scheduler and
  an optimizer. Many procedures are available at 
  `torch_uncertainty/optim_recipes.py <https://github.com/ENSTA-U2IS-AI/torch-uncertainty/blob/main/torch_uncertainty/optim_recipes.py>`_

* the path to the data and logs folder, in the example below, the root of the library
* and finally, the name of your model (used for logs)

Move to the directory containing your file and execute the code with :bash:`python3 experiment.py`.
Add lightning arguments such as :bash:`--accelerator gpu --devices "0, 1" --benchmark True` 
for multi-gpu training and cuDNN benchmark, etc.

Example
^^^^^^^

The following code - `available in the experiments folder <https://github.com/ENSTA-U2IS-AI/torch-uncertainty/blob/main/experiments/classification/cifar10/resnet.py>`_ - 
trains any ResNet architecture on CIFAR10:

.. code:: python

    from pathlib import Path

    from torch import nn

    from torch_uncertainty import cli_main, init_args
    from torch_uncertainty.baselines import ResNet
    from torch_uncertainty.datamodules import CIFAR10DataModule
    from torch_uncertainty.optim_recipes import get_procedure

    root = Path(__file__).parent.absolute().parents[1]

    args = init_args(ResNet, CIFAR10DataModule)

    net_name = f"{args.version}-resnet{args.arch}-cifar10"

    # datamodule
    args.root = str(root / "data")
    dm = CIFAR10DataModule(**vars(args))

    # model
    model = ResNet(
        num_classes=dm.num_classes,
        in_channels=dm.num_channels,
        loss=nn.CrossEntropyLoss(),
        optim_recipe=get_procedure(
            f"resnet{args.arch}", "cifar10", args.version
        ),
        style="cifar",
        **vars(args),
    )

    cli_main(model, dm, args.exp_dir, args.exp_name, args)

Run this model with, for instance:

.. code:: bash

    python3 resnet.py --version std --arch 18 --accelerator gpu --device 1 --benchmark True --max_epochs 75 --precision 16

You may replace the architecture (which should be a Lightning Module), the
Datamodule (a Lightning Datamodule), the loss or the optimization recipe to your likings.

Using the PyTorch-based models
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
backbone with the following code:

.. code:: python

    from torch_uncertainty.models.resnet import packed_resnet18

    model = packed_resnet18(
        in_channels = 3,
        num_estimators = 4,
        alpha = 2,
        gamma = 2,
        num_classes = 10,
    )

Using the PyTorch-based layers
------------------------------

Procedure
^^^^^^^^^

It is likely that your desired architecture is not supported by our library.
In that case, you might be interested in directly using the actual layers.

1. Check the API reference for specific layers of your choosing.
2. Import the layers and use them as you would for any standard PyTorch layer.

If you think that your architecture should be added to the package, raise an
issue on the GitHub repository!

.. tip::

  Do not hesitate to go to the API reference to get better explanations on the
  layer usage.

Example
^^^^^^^

You can create a Packed-Ensemble ``torch.nn.Module`` model with the following
code:

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
