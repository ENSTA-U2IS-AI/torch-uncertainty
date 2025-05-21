# ruff: noqa: F401
from .aggregated_dataset import AggregatedDataset
from .classification import (
    CIFAR10C,
    CIFAR10H,
    CIFAR10N,
    CIFAR100C,
    CIFAR100N,
    CUB,
    HTRU2,
    MNISTC,
    BankMarketing,
    DOTA2Games,
    ImageNetA,
    ImageNetC,
    ImageNetO,
    ImageNetR,
    NotMNIST,
    OnlineShoppers,
    OpenImageO,
    SpamBase,
    TinyImageNet,
    TinyImageNetC,
)
from .fractals import Fractals
from .frost import FrostImages
from .kitti import KITTIDepth
from .muad import MUAD
from .nyu import NYUv2
from .regression import UCIRegression
from .segmentation import CamVid, Cityscapes
