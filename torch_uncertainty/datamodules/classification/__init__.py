# ruff: noqa: F401
from .cifar10 import CIFAR10DataModule
from .cifar100 import CIFAR100DataModule
from .imagenet import ImageNetDataModule
from .mnist import MNISTDataModule
from .tiny_imagenet import TinyImageNetDataModule
from .uci import (
    BankMarketingDataModule,
    DOTA2GamesDataModule,
    HTRU2DataModule,
    OnlineShoppersDataModule,
    SpamBaseDataModule,
)
