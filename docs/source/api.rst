API Reference
=============

.. currentmodule:: torch_uncertainty

Routines
--------

The routine are the main building blocks of the library. They define the framework
in which the models are trained and evaluated. They allow for easy computation of different
metrics crucial for uncertainty estimation in different contexts, namely classification, regression and segmentation.

.. currentmodule:: torch_uncertainty.routines

Classification
^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    ClassificationRoutine

Segmentation
^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    SegmentationRoutine

Regression
^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    RegressionRoutine

Pixelwise Regression
^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    PixelRegressionRoutine

Baselines
---------

TorchUncertainty provide lightning-based models that can be easily trained and evaluated.
These models inherit from the routines and are specifically designed to benchmark
different methods in similar settings, here with constant architectures.

.. currentmodule:: torch_uncertainty.baselines.classification

Classification
^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    ResNetBaseline
    VGGBaseline
    WideResNetBaseline

.. currentmodule:: torch_uncertainty.baselines.regression

Regression
^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    MLPBaseline

.. currentmodule:: torch_uncertainty.baselines.segmentation

Segmentation
^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    DeepLabBaseline
    SegFormerBaseline

.. currentmodule:: torch_uncertainty.baselines.depth

Monocular Depth Estimation 
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    BTSBaseline

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
    LPBNNLinear
    LPBNNConv2d

Models
------

.. currentmodule:: torch_uncertainty.models

Wrappers
^^^^^^^^



Functions
"""""""""

.. autosummary::
    :toctree: generated/
    :nosignatures:

    deep_ensembles
    mc_dropout

Classes
"""""""

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    CheckpointEnsemble
    EMA
    MCDropout
    StochasticModel
    SWA
    SWAG

Metrics
-------

Classification
^^^^^^^^^^^^^^

.. currentmodule:: torch_uncertainty.metrics.classification

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    AURC
    AUSE
    FPR95
    AdaptiveCalibrationError
    BrierScore
    CalibrationError
    CategoricalNLL
    CovAt5Risk
    Disagreement
    Entropy
    GroupingLoss
    MeanIntersectionOverUnion
    MutualInformation
    RiskAt80Cov
    VariationRatio

Regression
^^^^^^^^^^

.. currentmodule:: torch_uncertainty.metrics.regression

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    DistributionNLL
    Log10
    MeanAbsoluteErrorInverse
    MeanGTRelativeAbsoluteError
    MeanGTRelativeSquaredError
    MeanSquaredErrorInverse
    MeanSquaredLogError
    SILog
    ThresholdAccuracy

Losses
------

.. currentmodule:: torch_uncertainty.losses

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    DistributionNLLLoss
    KLDiv
    ELBOLoss
    BetaNLL
    DECLoss

Post-Processing Methods
-----------------------

.. currentmodule:: torch_uncertainty.post_processing

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst
    
    MCBatchNorm
    LaplaceApprox
    
Scaling Methods
^^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class_inherited.rst

    TemperatureScaler
    VectorScaler
    MatrixScaler

Datamodules
-----------

.. currentmodule:: torch_uncertainty.datamodules.abstract

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst


.. currentmodule:: torch_uncertainty.datamodules

Classification
^^^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    CIFAR10DataModule
    CIFAR100DataModule
    MNISTDataModule
    TinyImageNetDataModule
    ImageNetDataModule

Regression
^^^^^^^^^^
.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    UCIDataModule

.. currentmodule:: torch_uncertainty.datamodules.segmentation

Segmentation
^^^^^^^^^^^^

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    CamVidDataModule
    CityscapesDataModule
    MUADDataModule

Datasets
--------

.. currentmodule:: torch_uncertainty.datasets

Classification
^^^^^^^^^^^^^^

.. currentmodule:: torch_uncertainty.datasets.classification

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    MNISTC
    NotMNIST
    CIFAR10C
    CIFAR100C
    CIFAR10H
    CIFAR10N
    CIFAR100N
    ImageNetA
    ImageNetC
    ImageNetO
    ImageNetR
    TinyImageNet
    TinyImageNetC
    OpenImageO

Regression
^^^^^^^^^^

.. currentmodule:: torch_uncertainty.datasets.regression

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    UCIRegression

Segmentation
^^^^^^^^^^^^

.. currentmodule:: torch_uncertainty.datasets.segmentation

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    CamVid
    Cityscapes

Others & Cross-Categories
^^^^^^^^^^^^^^^^^^^^^^^^^

.. currentmodule:: torch_uncertainty.datasets

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :template: class.rst

    Fractals
    FrostImages
    KITTIDepth
    MUAD
    NYUv2
