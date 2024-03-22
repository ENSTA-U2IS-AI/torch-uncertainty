# ruff: noqa: F401
from .classification.cifar10 import CIFAR10DataModule
from .classification.cifar100 import CIFAR100DataModule
from .classification.imagenet import ImageNetDataModule
from .classification.mnist import MNISTDataModule
from .classification.tiny_imagenet import TinyImageNetDataModule
from .segmentation import CamVidDataModule, CityscapesDataModule
from .uci_regression import UCIDataModule
