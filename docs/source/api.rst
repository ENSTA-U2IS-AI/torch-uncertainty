API reference
=============

.. currentmodule:: torch_uncertainty

Baselines
---------

.. currentmodule:: torch_uncertainty.baselines

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    PackedResNet

Models
------
.. currentmodule:: torch_uncertainty.models.resnet

**ResNet**

.. autosummary::
    :toctree: generated/
    :nosignatures:

    resnet18
    resnet34
    resnet50
    resnet101
    resnet152

**Packed-Ensembles ResNet**

.. autosummary::
    :toctree: generated/
    :nosignatures:

    packed_resnet18
    packed_resnet34
    packed_resnet50
    packed_resnet101
    packed_resnet152

**Masksembles ResNet**

.. autosummary::
    :toctree: generated/
    :nosignatures:

    masked_resnet18
    masked_resnet34
    masked_resnet50
    masked_resnet101
    masked_resnet152

Layers
------

.. currentmodule:: torch_uncertainty.layers

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    PackedConv2d
    PackedLinear

Metrics
-------

.. currentmodule:: torch_uncertainty.metrics

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    .. BrierScore
    Disagreement
    Entropy
    JensenShannonDivergence
    MutualInformation
    NegativeLogLikelihood