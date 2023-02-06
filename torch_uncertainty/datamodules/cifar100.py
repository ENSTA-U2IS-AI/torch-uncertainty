# fmt: off
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from timm.data.auto_augment import rand_augment_transform
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR100, SVHN

from ..datasets import CIFAR100_C, AggregatedDataset
from ..transforms import Cutout


# fmt: on
class CIFAR100DataModule(LightningDataModule):
    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int,
        val_split: int = 0,
        num_workers: int = 1,
        enable_cutout: bool = False,
        enable_randaugment: bool = False,
        auto_augment: str = None,
        use_cifar_c: str = None,
        corrupution_severity: int = 1,
        num_dataloaders: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(root, str):
            root = Path(root)

        self.root: Path = root
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.enable_cutout = enable_cutout
        self.enable_randaugment = enable_randaugment
        self.auto_augment = auto_augment
        self.num_dataloaders = num_dataloaders
        self.num_classes = 100
        self.num_channels = 3

        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        if use_cifar_c is None:
            self.dataset = CIFAR100
        else:
            self.dataset = CIFAR100_C

        self.ood_dataset = SVHN

        self.use_cifar_c = use_cifar_c
        self.corruption_severity = corrupution_severity

        assert (
            self.enable_cutout
            + self.enable_randaugment
            + int(self.auto_augment is not None)
            <= 1
        ), "Only one data augmentation can be chosen at a time."

        if enable_cutout:
            main_transform = Cutout(8)
        elif enable_randaugment:
            main_transform = T.RandAugment(num_ops=2, magnitude=20)
        elif auto_augment:
            main_transform = rand_augment_transform(self.auto_augment, {})
        else:
            main_transform = nn.Identity()

        self.transform_train = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                main_transform,
                T.ToTensor(),
                T.ConvertImageDtype(torch.float32),
                T.Normalize(
                    (0.5071, 0.4867, 0.4408),
                    (0.2675, 0.2565, 0.2761),
                ),
            ]
        )
        self.transform_test = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    (0.5071, 0.4867, 0.4408),
                    (0.2675, 0.2565, 0.2761),
                ),
            ]
        )

    def prepare_data(self) -> None:
        if self.use_cifar_c is None:
            self.dataset(self.root, train=True, download=True)
            self.dataset(self.root, train=False, download=True)

        self.ood_dataset(
            self.root,
            split="test",
            download=True,
            transform=self.transform_test,
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            assert (
                self.use_cifar_c is None
            ), "CIFAR-C can only be used in testing."
            full = self.dataset(
                self.root,
                train=True,
                download=False,
                transform=self.transform_train,
            )
            self.train, self.val = random_split(
                full, [len(full) - self.val_split, self.val_split]
            )
            if self.val_split == 0:
                self.val = self.dataset(
                    self.root,
                    train=False,
                    download=False,
                    transform=self.transform_test,
                )
        elif stage == "test":
            if self.use_cifar_c is None:
                self.test = self.dataset(
                    self.root,
                    train=False,
                    download=False,
                    transform=self.transform_test,
                )
            else:
                self.test = self.dataset(
                    self.root,
                    transform=self.transform_test,
                    subset=self.use_cifar_c,
                    severity=self.corruption_severity,
                )
            self.ood = self.ood_dataset(
                self.root,
                split="test",
                download=False,
                transform=self.transform_test,
            )

    def train_dataloader(self) -> DataLoader:
        r"""Gets the training dataloader for CIFAR10.
        Returns:
            DataLoader: CIFAR10 training dataloader.
        """
        if self.num_dataloaders > 1:
            return self._data_loader(
                AggregatedDataset(self.train, self.num_dataloaders),
                shuffle=True,
            )
        else:
            return self._data_loader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        r"""Gets the validation dataloader for CIFAR100.
        Returns:
            DataLoader: CIFAR100 validation dataloader.
        """
        return self._data_loader(self.val)

    def test_dataloader(self) -> List[DataLoader]:
        r"""Gets the test dataloaders for CIFAR100.
        Returns:
            List[DataLoader]: Dataloaders of the CIFAR100 test set (in
                distribution data) and SVHN test split (out-of-distribution
                data).
        """
        return [self._data_loader(self.test), self._data_loader(self.ood)]

    def _data_loader(
        self, dataset: Dataset, shuffle: bool = False
    ) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    @classmethod
    def add_argparse_args(
        cls,
        parent_parser: ArgumentParser,
        **kwargs: Any,
    ) -> ArgumentParser:
        p = parent_parser.add_argument_group("datamodule")
        p.add_argument("--root", type=str, default="./data/")
        p.add_argument("--batch_size", type=int, default=128)
        p.add_argument("--val_split", type=int, default=0)
        p.add_argument("--num_workers", type=int, default=4)
        p.add_argument("--cutout", dest="enable_cutout", action="store_true")
        p.add_argument(
            "--randaugment", dest="enable_randaugment", action="store_true"
        )
        p.add_argument("--auto_augment", type=str)
        p.add_argument("--cifar-c", dest="use_cifar_c", type=str, default=None)
        p.add_argument(
            "--severity", dest="corrupution_severity", type=int, default=1
        )
        return parent_parser
