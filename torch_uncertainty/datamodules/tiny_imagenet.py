from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Literal

import torchvision.transforms as T
from numpy.typing import ArrayLike
from timm.data.auto_augment import rand_augment_transform
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader
from torchvision.datasets import DTD, SVHN

from torch_uncertainty.datasets.classification import ImageNetO, TinyImageNet

from .abstract import AbstractDataModule


class TinyImageNetDataModule(AbstractDataModule):
    num_classes = 200
    num_channels = 3
    training_task = "classification"

    def __init__(
        self,
        root: str | Path,
        eval_ood: bool,
        batch_size: int,
        ood_ds: str = "svhn",
        rand_augment_opt: str | None = None,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            root=root,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        # TODO: COMPUTE STATS

        self.eval_ood = eval_ood
        self.ood_ds = ood_ds

        self.dataset = TinyImageNet

        if ood_ds == "imagenet-o":
            self.ood_dataset = ImageNetO
        elif ood_ds == "svhn":
            self.ood_dataset = SVHN
        elif ood_ds == "textures":
            self.ood_dataset = DTD
        else:
            raise ValueError(
                f"OOD dataset {ood_ds} not supported for TinyImageNet."
            )

        if rand_augment_opt is not None:
            main_transform = rand_augment_transform(rand_augment_opt, {})
        else:
            main_transform = nn.Identity()

        self.train_transform = T.Compose(
            [
                T.RandomCrop(64, padding=4),
                T.RandomHorizontalFlip(),
                main_transform,
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.test_transform = T.Compose(
            [
                T.Resize(64),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def _verify_splits(self, split: str) -> None:  # coverage: ignore
        if split not in list(self.root.iterdir()):
            raise FileNotFoundError(
                f"a {split} TinyImagenet split was not found in {self.root},"
                f" make sure the folder contains a subfolder named {split}"
            )

    def prepare_data(self) -> None:  # coverage: ignore
        if self.eval_ood:
            if self.ood_ds != "textures":
                self.ood_dataset(
                    self.root,
                    split="test",
                    download=True,
                    transform=self.test_transform,
                )
            else:
                ConcatDataset(
                    [
                        self.ood_dataset(
                            self.root,
                            split="train",
                            download=True,
                            transform=self.test_transform,
                        ),
                        self.ood_dataset(
                            self.root,
                            split="val",
                            download=True,
                            transform=self.test_transform,
                        ),
                        self.ood_dataset(
                            self.root,
                            split="test",
                            download=True,
                            transform=self.test_transform,
                        ),
                    ]
                )

    def setup(self, stage: Literal["fit", "test"] | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train = self.dataset(
                self.root,
                split="train",
                transform=self.train_transform,
            )
            self.val = self.dataset(
                self.root,
                split="val",
                transform=self.test_transform,
            )
        elif stage == "test":
            self.test = self.dataset(
                self.root,
                split="val",
                transform=self.test_transform,
            )
        else:
            raise ValueError(f"Stage {stage} is not supported.")

        if self.eval_ood:
            if self.ood_ds == "textures":
                self.ood = ConcatDataset(
                    [
                        self.ood_dataset(
                            self.root,
                            split="train",
                            download=True,
                            transform=self.test_transform,
                        ),
                        self.ood_dataset(
                            self.root,
                            split="val",
                            download=True,
                            transform=self.test_transform,
                        ),
                        self.ood_dataset(
                            self.root,
                            split="test",
                            download=True,
                            transform=self.test_transform,
                        ),
                    ]
                )
            else:
                self.ood = self.ood_dataset(
                    self.root,
                    split="test",
                    transform=self.test_transform,
                )

    def train_dataloader(self) -> DataLoader:
        r"""Get the training dataloader for TinyImageNet.

        Return:
            DataLoader: TinyImageNet training dataloader.
        """
        return self._data_loader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        r"""Get the validation dataloader for TinyImageNet.

        Return:
            DataLoader: TinyImageNet validation dataloader.
        """
        return self._data_loader(self.val)

    def test_dataloader(self) -> list[DataLoader]:
        r"""Get test dataloaders for TinyImageNet.

        Return:
            List[DataLoader]: test set for in distribution data
            and out-of-distribution data.
        """
        dataloader = [self._data_loader(self.test)]
        if self.eval_ood:
            dataloader.append(self._data_loader(self.ood))
        return dataloader

    def _get_train_data(self) -> ArrayLike:
        return self.train.samples

    def _get_train_targets(self) -> ArrayLike:
        return self.train.label_data

    @classmethod
    def add_argparse_args(
        cls,
        parent_parser: ArgumentParser,
        **kwargs: Any,
    ) -> ArgumentParser:
        p = super().add_argparse_args(parent_parser)

        # Arguments for Tiny Imagenet
        p.add_argument(
            "--rand_augment", dest="rand_augment_opt", type=str, default=None
        )
        p.add_argument("--eval-ood", action="store_true")
        return parent_parser
