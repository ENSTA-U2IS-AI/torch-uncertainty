# fmt: off
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

import torch
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from timm.data.auto_augment import rand_augment_transform
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import CIFAR100, SVHN

from ..datasets import AggregatedDataset
from ..datasets.classification import CIFAR100C
from ..transforms import Cutout


# fmt: on
class CIFAR100DataModule(LightningDataModule):
    """DataModule for CIFAR100.

    Args:
        root (str): Root directory of the datasets.
        batch_size (int): Number of samples per batch.
        val_split (float): Share of samples to use for validation. Defaults
            to ``0.0``.
        num_workers (int): Number of workers to use for data loading. Defaults
            to ``1``.
        cutout (int): Size of cutout to apply to images. Defaults to ``None``.
        randaugment (bool): Whether to apply RandAugment. Defaults to
            ``False``.
        auto_augment (str): Which auto-augment to apply. Defaults to ``None``.
        test_alt (str): Which test set to use. Defaults to ``None``.
        corruption_severity (int): Severity of corruption to apply to
            CIFAR100-C. Defaults to ``1``.
        num_dataloaders (int): Number of dataloaders to use. Defaults to ``1``.
        pin_memory (bool): Whether to pin memory. Defaults to ``True``.
        persistent_workers (bool): Whether to use persistent workers. Defaults
            to ``True``.
    """

    num_classes = 100
    num_channels = 3
    input_shape = (3, 32, 32)
    training_task = "classification"

    def __init__(
        self,
        root: Union[str, Path],
        ood_detection: bool,
        batch_size: int,
        val_split: float = 0.0,
        num_workers: int = 1,
        cutout: Optional[int] = None,
        randaugment: bool = False,
        auto_augment: Optional[str] = None,
        test_alt: Optional[Literal["c"]] = None,
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
        self.ood_detection = ood_detection
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.num_dataloaders = num_dataloaders

        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers

        if test_alt == "c":
            self.dataset = CIFAR100C
        else:
            self.dataset = CIFAR100

        self.test_alt = test_alt

        self.ood_dataset = SVHN

        self.corruption_severity = corruption_severity

        if (cutout is not None) + randaugment + int(
            auto_augment is not None
        ) > 1:
            raise ValueError(
                "Only one data augmentation can be chosen at a time. Raise a "
                "GitHub issue if needed."
            )

        if cutout:
            main_transform = Cutout(cutout)
        elif randaugment:
            main_transform = T.RandAugment(num_ops=2, magnitude=20)
        elif auto_augment:
            main_transform = rand_augment_transform(auto_augment, {})
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

    def prepare_data(self) -> None:  # coverage: ignore
        if self.test_alt is None:
            self.dataset(self.root, train=True, download=True)
            self.dataset(self.root, train=False, download=True)

        if self.ood_detection:
            self.ood_dataset(
                self.root,
                split="test",
                download=True,
                transform=self.transform_test,
            )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            if self.test_alt == "c":
                raise ValueError("CIFAR-C can only be used in testing.")
            full = self.dataset(
                self.root,
                train=True,
                download=False,
                transform=self.transform_train,
            )
            if self.val_split:
                self.train, self.val = random_split(
                    full,
                    [
                        1 - self.val_split,
                        self.val_split,
                    ],
                )
            else:
                self.train = full
                self.val = self.dataset(
                    self.root,
                    train=False,
                    download=False,
                    transform=self.transform_test,
                )
        elif stage == "test":
            if self.test_alt is None:
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
                    severity=self.corruption_severity,
                )
            if self.ood_detection:
                self.ood = self.ood_dataset(
                    self.root,
                    split="test",
                    download=False,
                    transform=self.transform_test,
                )
        else:
            raise ValueError(f"Stage {stage} is not supported.")

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader for CIFAR100.

        Return:
            DataLoader: CIFAR100 training dataloader.
        """
        if self.num_dataloaders > 1:
            return self._data_loader(
                AggregatedDataset(self.train, self.num_dataloaders),
                shuffle=True,
            )
        else:
            return self._data_loader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader for CIFAR100.

        Return:
            DataLoader: CIFAR100 validation dataloader.
        """
        return self._data_loader(self.val)

    def test_dataloader(self) -> List[DataLoader]:
        """Get the test dataloaders for CIFAR100.

        Return:
            List[DataLoader]: Dataloaders of the CIFAR100 test set (in
                distribution data) and SVHN test split (out-of-distribution
                data).
        """
        dataloader = [self._data_loader(self.test)]
        if self.ood_detection:
            dataloader.append(self._data_loader(self.ood))
        return dataloader

    def _data_loader(
        self, dataset: Dataset, shuffle: bool = False
    ) -> DataLoader:
        """Create a dataloader for a given dataset.

        Args:
            dataset (Dataset): Dataset to create a dataloader for.
            shuffle (bool, optional): Whether to shuffle the dataset. Defaults
                to False.

        Return:
            DataLoader: Dataloader for the given dataset.
        """
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
        p.add_argument("--val_split", type=float, default=0.0)
        p.add_argument("--num_workers", type=int, default=4)
        p.add_argument(
            "--evaluate_ood", dest="ood_detection", action="store_true"
        )
        p.add_argument("--cutout", type=int, default=0)
        p.add_argument("--randaugment", dest="randaugment", action="store_true")
        p.add_argument("--auto_augment", type=str)
        p.add_argument("--test_alt", choices=["c"], default=None)
        p.add_argument(
            "--severity", dest="corruption_severity", type=int, default=1
        )
        return parent_parser
