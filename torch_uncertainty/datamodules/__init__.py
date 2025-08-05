# ruff: noqa: F401
from .abstract import TUDataModule
from .classification import (
    BankMarketingDataModule,
    CIFAR10DataModule,
    CIFAR100DataModule,
    DOTA2GamesDataModule,
    HTRU2DataModule,
    ImageNet200DataModule,
    ImageNetDataModule,
    MNISTDataModule,
    OnlineShoppersDataModule,
    SpamBaseDataModule,
    Sst2DataModule,
    TinyImageNetDataModule,
    UCIClassificationDataModule,
    UCRUEADataModule,
)
from .segmentation import CamVidDataModule, CityscapesDataModule, MUADDataModule
from .uci_regression import UCIRegressionDataModule
