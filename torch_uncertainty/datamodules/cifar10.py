# fmt: off
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional, Union

import torch.nn as nn
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from timm.data.auto_augment import rand_augment_transform
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR10, SVHN

from ..datasets import CIFAR10_C, AggregatedDataset
from ..transforms import Cutout


# fmt: on
class CIFAR10DataModule(LightningDataModule):
    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int,
        val_split: int = 0,
        num_workers: int = 1,
        enable_cutout: bool = False,
        auto_augment: str = None,
        use_cifar_c: str = None,
        corruption_severity: int = 1,
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
        self.auto_augment = auto_augment
        self.num_dataloaders = num_dataloaders
        self.num_classes = 10
        self.num_channels = 3

        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        if use_cifar_c is None:
            self.dataset = CIFAR10
        else:
            self.dataset = CIFAR10_C

        self.use_cifar_c = use_cifar_c
        self.corruption_severity = corruption_severity
        self.ood_dataset = SVHN

        if enable_cutout:
            main_transform = Cutout(16)
        elif auto_augment:
            main_transform = rand_augment_transform(self.auto_augment, {})
        else:
            main_transform = nn.Identity()

        self.transform_train = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
                main_transform,
            ]
        )

        self.transform_test = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ]
        )

        self.transform_test_imagenet = T.Compose(
            [
                T.ToTensor(),
                T.RandomCrop(32),
                T.RandomHorizontalFlip(),
                T.Normalize(
                    (0.4914, 0.4822, 0.4465),
                    (0.2023, 0.1994, 0.2010),
                ),
            ]
        )

    def prepare_data(self) -> None:
        if self.use_cifar_c is None:
            self.dataset(self.root, train=True, download=True)
            self.dataset(self.root, train=False, download=True)
        else:
            self.dataset(
                self.root,
                subset=self.use_cifar_c,
                severity=self.corruption_severity,
            )

        # if self.use_imagenet_o:
        #     self.ood_dataset(self.root)
        # else:
        self.ood_dataset(self.root, split="test", download=True)

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
        elif stage == "test" or stage is None:
            self.test = self.dataset(
                self.root,
                train=False,
                download=False,
                transform=self.transform_test,
            )
            # if self.use_imagenet_o:
            #     self.ood = self.ood_dataset(
            #         self.root,
            #         transform=self.transform_test_imagenet,
            #     )
            # else:
            self.ood = self.ood_dataset(
                self.root,
                split="test",
                download=False,
                transform=self.transform_test,
            )
        else:
            self.test = self.dataset(
                self.root,
                subset=self.use_cifar_c,
                severity=self.corruption_severity,
                transform=self.transform_test,
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
        r"""Gets the validation dataloader for CIFAR10.
        Returns:
            DataLoader: CIFAR10 validation dataloader.
        """
        return self._data_loader(self.val)

    def predict_dataloader(self) -> DataLoader:
        r"""Gets the validation dataloader for CIFAR10.
        Returns:
            DataLoader: CIFAR10 validation dataloader.
        """
        return self._data_loader(self.val)

    def test_dataloader(self) -> List[DataLoader]:
        r"""Gets the test dataloaders for CIFAR10.
        Returns:
            List[DataLoader]: Dataloaders of the CIFAR10 test set (in
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
        p.add_argument("--auto_augment", type=str)
        # p.add_argument(
        #     "--imagenet-o", dest="use_imagenet_o", action="store_true"
        # )
        p.add_argument("--cifar-c", dest="use_cifar_c", type=str, default=None)
        p.add_argument(
            "--severity", dest="corruption_severity", type=int, default=None
        )
        return parent_parser
