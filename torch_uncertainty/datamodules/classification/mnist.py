from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import v2

from torch_uncertainty.datamodules import TUDataModule
from torch_uncertainty.datasets.classification import MNISTC, NotMNIST
from torch_uncertainty.transforms import Cutout
from torch_uncertainty.utils import create_train_val_split


class MNISTDataModule(TUDataModule):
    num_classes = 10
    num_channels = 1
    input_shape = (1, 28, 28)
    training_task = "classification"
    ood_datasets = ["fashion", "notMNIST"]
    mean = (0.1307,)
    std = (0.3081,)

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        eval_ood: bool = False,
        eval_shift: bool = False,
        ood_ds: Literal["fashion", "notMNIST"] = "fashion",
        val_split: float | None = None,
        postprocess_set: Literal["val", "test"] = "val",
        num_workers: int = 1,
        basic_augment: bool = True,
        cutout: int | None = None,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        """DataModule for MNIST.

        Args:
            root (str): Root directory of the datasets.
            eval_ood (bool): Whether to evaluate on out-of-distribution data.
                Defaults to ``False``.
            eval_shift (bool): Whether to evaluate on shifted data. Defaults to
                ``False``.
            batch_size (int): Number of samples per batch.
            ood_ds (str): Which out-of-distribution dataset to use. Defaults to
                ``"fashion"``; `fashion` stands for FashionMNIST and `notMNIST` for
                notMNIST.
            val_split (float): Share of samples to use for validation. Defaults
                to ``0.0``.
            postprocess_set (str, optional): The post-hoc calibration dataset to
                use for the post-processing method. Defaults to ``val``.
            num_workers (int): Number of workers to use for data loading. Defaults
                to ``1``.
            basic_augment (bool): Whether to apply base augmentations. Defaults to
                ``True``.
            cutout (int): Size of cutout to apply to images. Defaults to ``None``.
            pin_memory (bool): Whether to pin memory. Defaults to ``True``.
            persistent_workers (bool): Whether to use persistent workers. Defaults
                to ``True``.
        """
        super().__init__(
            root=root,
            batch_size=batch_size,
            val_split=val_split,
            postprocess_set=postprocess_set,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.eval_ood = eval_ood
        self.eval_shift = eval_shift
        self.batch_size = batch_size

        self.dataset = MNIST

        if ood_ds == "fashion":
            self.ood_dataset = FashionMNIST
        elif ood_ds == "notMNIST":
            self.ood_dataset = NotMNIST
        else:
            raise ValueError(f"`ood_ds` should be in {self.ood_datasets}. Got {ood_ds}.")
        self.shift_dataset = MNISTC
        self.shift_severity = 1

        basic_transform = v2.RandomCrop(28, padding=4) if basic_augment else nn.Identity()

        main_transform = Cutout(cutout) if cutout else nn.Identity()

        self.train_transform = v2.Compose(
            [
                v2.ToImage(),
                basic_transform,
                main_transform,
                v2.ToDtype(dtype=torch.float32, scale=True),
                v2.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self.test_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.CenterCrop(28),
                v2.ToDtype(dtype=torch.float32, scale=True),
                v2.Normalize(mean=self.mean, std=self.std),
            ]
        )
        if self.eval_ood:  # NotMNIST has 3 channels
            self.ood_transform = v2.Compose(
                [
                    v2.ToImage(),
                    v2.Grayscale(num_output_channels=1),
                    v2.CenterCrop(28),
                    v2.ToDtype(dtype=torch.float32, scale=True),
                    v2.Normalize(mean=self.mean, std=self.std),
                ]
            )

    def prepare_data(self) -> None:  # coverage: ignore
        """Download the datasets."""
        self.dataset(self.root, train=True, download=True)
        self.dataset(self.root, train=False, download=True)

        if self.eval_ood:
            self.ood_dataset(self.root, download=True)
        if self.eval_shift:
            self.shift_dataset(self.root, download=True)

    def setup(self, stage: Literal["fit", "test"] | None = None) -> None:
        if stage == "fit" or stage is None:
            full = self.dataset(
                self.root,
                train=True,
                download=False,
                transform=self.train_transform,
            )
            if self.val_split:
                self.train, self.val = create_train_val_split(
                    full,
                    self.val_split,
                    self.test_transform,
                )
            else:
                self.train = full
                self.val = self.dataset(
                    self.root,
                    train=False,
                    download=False,
                    transform=self.test_transform,
                )
        if stage == "test" or stage is None:
            self.test = self.dataset(
                self.root,
                train=False,
                download=False,
                transform=self.test_transform,
            )
        if stage not in ["fit", "test", None]:
            raise ValueError(f"Stage {stage} is not supported.")

        if self.eval_ood:
            self.ood = self.ood_dataset(
                self.root,
                download=False,
                transform=self.ood_transform,
            )
        if self.eval_shift:
            self.shift = self.shift_dataset(
                self.root,
                download=False,
                transform=self.test_transform,
            )

    def test_dataloader(self) -> list[DataLoader]:
        r"""Get the test dataloaders for MNIST.

        Return:
            list[DataLoader]: Dataloaders of the MNIST test set (in
                distribution data), FashionMNIST or NotMNIST test split
                (out-of-distribution data), and/or MNISTC (shifted data).
        """
        dataloader = [self._data_loader(self.test)]
        if self.eval_ood:
            dataloader.append(self._data_loader(self.ood))
        if self.eval_shift:
            dataloader.append(self._data_loader(self.shift))
        return dataloader
