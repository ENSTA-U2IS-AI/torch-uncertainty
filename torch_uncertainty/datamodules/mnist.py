from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Literal

import torchvision.transforms as T
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST, FashionMNIST

from torch_uncertainty.datamodules.abstract import AbstractDataModule
from torch_uncertainty.datasets.classification import MNISTC, NotMNIST
from torch_uncertainty.transforms import Cutout


class MNISTDataModule(AbstractDataModule):
    num_classes = 10
    num_channels = 1
    input_shape = (1, 28, 28)
    training_task = "classification"
    ood_datasets = ["fashion", "not"]

    def __init__(
        self,
        root: str | Path,
        eval_ood: bool,
        batch_size: int,
        ood_ds: Literal["fashion", "not"] = "fashion",
        val_split: float = 0.0,
        num_workers: int = 1,
        cutout: int | None = None,
        test_alt: Literal["c"] | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs,
    ) -> None:
        """DataModule for MNIST.

        Args:
            root (str): Root directory of the datasets.
            eval_ood (bool): Whether to evaluate on out-of-distribution data.
            batch_size (int): Number of samples per batch.
            ood_ds (str): Which out-of-distribution dataset to use. Defaults to
                ``"fashion"``; `fashion` stands for FashionMNIST and `not` for
                notMNIST.
            val_split (float): Share of samples to use for validation. Defaults
                to ``0.0``.
            num_workers (int): Number of workers to use for data loading. Defaults
                to ``1``.
            cutout (int): Size of cutout to apply to images. Defaults to ``None``.
            test_alt (str): Which test set to use. Defaults to ``None``.
            pin_memory (bool): Whether to pin memory. Defaults to ``True``.
            persistent_workers (bool): Whether to use persistent workers. Defaults
                to ``True``.
            kwargs: Additional arguments.
        """
        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.eval_ood = eval_ood
        self.batch_size = batch_size
        self.val_split = val_split

        if test_alt == "c":
            self.dataset = MNISTC
        else:
            self.dataset = MNIST

        if ood_ds == "fashion":
            self.ood_dataset = FashionMNIST
        elif ood_ds == "not":
            self.ood_dataset = NotMNIST
        else:
            raise ValueError(
                f"`ood_ds` should be `fashion` or `not`. Got {ood_ds}."
            )

        main_transform = Cutout(cutout) if cutout else nn.Identity()

        self.train_transform = T.Compose(
            [
                main_transform,
                T.ToTensor(),
                T.RandomCrop(28, padding=4),
                T.Normalize((0.1307,), (0.3081,)),
            ]
        )
        self.test_transform = T.Compose(
            [
                T.ToTensor(),
                T.CenterCrop(28),
                T.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self) -> None:  # coverage: ignore
        """Download the datasets."""
        self.dataset(self.root, train=True, download=True)
        self.dataset(self.root, train=False, download=True)

        if self.eval_ood:
            self.ood_dataset(self.root, download=True)

    def setup(self, stage: Literal["fit", "test"] | None = None) -> None:
        if stage == "fit" or stage is None:
            full = self.dataset(
                self.root,
                train=True,
                download=False,
                transform=self.train_transform,
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
                    transform=self.test_transform,
                )
        elif stage == "test":
            self.test = self.dataset(
                self.root,
                train=False,
                download=False,
                transform=self.test_transform,
            )
        else:
            raise ValueError(f"Stage {stage} is not supported.")

        if self.eval_ood:
            self.ood = self.ood_dataset(
                self.root,
                download=False,
                transform=self.test_transform,
            )

    def test_dataloader(self) -> list[DataLoader]:
        r"""Get the test dataloaders for MNIST.

        Return:
            List[DataLoader]: Dataloaders of the MNIST test set (in
                distribution data) and FashionMNIST test split
                (out-of-distribution data).
        """
        dataloader = [self._data_loader(self.test)]
        if self.eval_ood:
            dataloader.append(self._data_loader(self.ood))
        return dataloader

    @classmethod
    def add_argparse_args(
        cls,
        parent_parser: ArgumentParser,
        **kwargs: Any,
    ) -> ArgumentParser:
        p = super().add_argparse_args(parent_parser)

        # Arguments for MNIST
        p.add_argument("--eval-ood", action="store_true")
        p.add_argument("--ood_ds", choices=cls.ood_datasets, default="fashion")
        p.add_argument("--test_alt", choices=["c"], default=None)
        return parent_parser
