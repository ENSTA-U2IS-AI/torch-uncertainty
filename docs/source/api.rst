API reference
=============

.. currentmodule:: torch_uncertainty

Baselines
---------

This API provides lightning-based models that can be easily trained and evaluated.

.. currentmodule:: torch_uncertainty.baselines.classification

Classification
^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    ResNet
    VGG
    WideResNet

.. currentmodule:: torch_uncertainty.baselines.regression

Regression
^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    MLP

.. Models
.. ------

.. This section encapsulates the model backbones currently supported by the library.

.. ResNet
.. ^^^^^^

.. .. currentmodule:: torch_uncertainty.models.resnet

.. Concerning ResNet backbones, we provide building functions for ResNet18, ResNet34,
.. ResNet50, ResNet101 and, ResNet152 (from `Deep Residual Learning for Image Recognition
.. <https://arxiv.org/pdf/1512.03385.pdf>`_, CVPR 2016). 

.. Standard
.. ~~~~~~~

.. .. autosummary::
..     :toctree: generated/
..     :nosignatures:

..     resnet18
..     resnet34
..     resnet50
..     resnet101
..     resnet152

.. Packed-Ensembles
.. ~~~~~~~~~~~~~~~~

.. .. autosummary::
..     :toctree: generated/
..     :nosignatures:

..     packed_resnet18
..     packed_resnet34
..     packed_resnet50
..     packed_resnet101
..     packed_resnet152

.. Masksembles
.. ~~~~~~~~~~~

.. .. autosummary::
..     :toctree: generated/
..     :nosignatures:

..     masked_resnet18
..     masked_resnet34
..     masked_resnet50
..     masked_resnet101
..     masked_resnet152

.. BatchEnsemble
.. ~~~~~~~~~~~~~

.. .. autosummary::
..     :toctree: generated/
..     :nosignatures:

..     batched_resnet18
..     batched_resnet34
..     batched_resnet50
..     batched_resnet101
..     batched_resnet152

.. Wide-ResNet
.. ^^^^^^^^^^^

.. .. currentmodule:: torch_uncertainty.models.wideresnet

.. Concerning Wide-ResNet backbones, we provide building functions for Wide-ResNet28x10
.. (from `Wide Residual Networks <https://arxiv.org/pdf/1605.07146.pdf>`_, British
.. Machine Vision Conference 2016).

.. Standard
.. ~~~~~~~

.. .. autosummary::
..     :toctree: generated/
..     :nosignatures:

..     wideresnet28x10

.. Packed-Ensembles
.. ~~~~~~~~~~~~~~~~

.. .. autosummary::
..     :toctree: generated/
..     :nosignatures:

..     packed_wideresnet28x10

.. Masksembles
.. ~~~~~~~~~~~

.. .. autosummary::
..     :toctree: generated/
..     :nosignatures:

..     masked_wideresnet28x10

.. BatchEnsemble
.. ~~~~~~~~~~~~~

.. .. autosummary::
..     :toctree: generated/
..     :nosignatures:

..     batched_wideresnet28x10

Layers
------

Ensemble layers
^^^^^^^^^^^^^^^

.. currentmodule:: torch_uncertainty.layers

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    PackedLinear
    PackedConv2d
    BatchLinear
    BatchConv2d
    MaskedLinear
    MaskedConv2d


Bayesian layers
^^^^^^^^^^^^^^^

.. currentmodule:: torch_uncertainty.layers.bayesian

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    BayesLinear
    BayesConv1d
    BayesConv2d
    BayesConv3d

Metrics
-------

.. currentmodule:: torch_uncertainty.metrics

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    AUSE
    BrierScore
    CE
    Disagreement
    Entropy
    MutualInformation
    NegativeLogLikelihood
    GaussianNegativeLogLikelihood
    FPR95

Losses
------

.. currentmodule:: torch_uncertainty.losses

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    KLDiv
    ELBOLoss
    BetaNLL
    NIGLoss
    DECLoss

Post-Processing Methods
-----------------------

.. currentmodule:: torch_uncertainty.post_processing

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class_inherited.rst

    TemperatureScaler
    VectorScaler
    MatrixScaler

Datamodules
-----------

.. currentmodule:: torch_uncertainty.datamodules

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    CIFAR10DataModule
    CIFAR100DataModule
    MNISTDataModule
    TinyImageNetDataModule
    ImageNetDataModule
    UCIDataModule
