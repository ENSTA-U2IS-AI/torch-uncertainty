from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Literal

import torchvision.transforms as T
from timm.data.auto_augment import rand_augment_transform
from timm.data.mixup import Mixup
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import DTD, SVHN, ImageNet, INaturalist

from torch_uncertainty.datamodules.abstract import AbstractDataModule
from torch_uncertainty.datasets.classification import (
    ImageNetA,
    ImageNetO,
    ImageNetR,
)


class ImageNetDataModule(AbstractDataModule):
    num_classes = 1000
    num_channels = 3
    test_datasets = ["r", "o", "a"]
    ood_datasets = ["inaturalist", "imagenet-o", "svhn", "textures"]
    training_task = "classification"

    def __init__(
        self,
        root: str | Path,
        evaluate_ood: bool,
        batch_size: int,
        ood_ds: str = "svhn",
        test_alt: str | None = None,
        procedure: str = "A3",
        train_size: int = 224,
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

        self.evaluate_ood = evaluate_ood
        self.ood_ds = ood_ds
        self.test_alt = test_alt

        if test_alt is None:
            self.dataset = ImageNet
        elif test_alt == "r":
            self.dataset = ImageNetR
        elif test_alt == "o":
            self.dataset = ImageNetO
        elif test_alt == "a":
            self.dataset = ImageNetA
        else:
            raise ValueError(f"The alternative {test_alt} is not known.")

        if ood_ds == "inaturalist":
            self.ood_dataset = INaturalist
        elif ood_ds == "imagenet-o":
            self.ood_dataset = ImageNetO
        elif ood_ds == "svhn":
            self.ood_dataset = SVHN
        elif ood_ds == "textures":
            self.ood_dataset = DTD
        else:
            raise ValueError(f"The dataset {ood_ds} is not supported.")

        self.procedure = procedure

        if self.procedure is None:
            print("Custom Procedure")
            if rand_augment_opt is not None:
                main_transform = rand_augment_transform(rand_augment_opt, {})
            else:
                main_transform = nn.Identity()
        elif self.procedure == "ViT":
            train_size = 224
            main_transform = T.Compose(
                [
                    Mixup(mixup_alpha=0.2, cutmix_alpha=1.0),
                    rand_augment_transform("rand-m9-n2-mstd0.5", {}),
                ]
            )

        elif self.procedure == "A3":
            print("Procedure A3")
            train_size = 160
            main_transform = rand_augment_transform("rand-m6-mstd0.5-inc1", {})
        else:
            raise ValueError("The procedure is unknown")

        self.transform_train = T.Compose(
            [
                T.RandomResizedCrop(train_size),
                T.RandomHorizontalFlip(),
                main_transform,
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

        self.transform_test = T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    def _verify_splits(self, split: str) -> None:
        if split not in list(self.root.iterdir()):
            raise FileNotFoundError(
                f"a {split} Imagenet split was not found in {self.root},"
                f" make sure the folder contains a subfolder named {split}"
            )

    def prepare_data(self) -> None:  # coverage: ignore
        if self.test_alt is not None:
            self.data = self.dataset(
                self.root,
                split="val",
                download=True,
            )
        if self.evaluate_ood:
            if self.ood_ds == "inaturalist":
                self.ood = self.ood_dataset(
                    self.root,
                    version="2021_valid",
                    download=True,
                    transform=self.transform_test,
                )
            elif self.ood_ds != "textures":
                self.ood = self.ood_dataset(
                    self.root,
                    split="test",
                    download=True,
                    transform=self.transform_test,
                )
            else:
                self.ood = self.ood_dataset(
                    self.root,
                    split="train",
                    download=True,
                    transform=self.transform_test,
                )

    def setup(self, stage: Literal["fit", "test"] | None = None) -> None:
        if stage == "fit" or stage is None:
            if self.test_alt is not None:
                raise ValueError(
                    "The test_alt argument is not supported for training."
                )
            self.train = self.dataset(
                self.root,
                split="train",
                transform=self.transform_train,
            )
            self.val = self.dataset(
                self.root,
                split="val",
                transform=self.transform_test,
            )
        elif stage == "test":
            self.test = self.dataset(
                self.root,
                split="val",
                transform=self.transform_test,
            )
        else:
            raise ValueError(f"Stage {stage} is not supported.")

        if self.evaluate_ood:
            if self.ood_ds == "inaturalist":
                self.ood = self.ood_dataset(
                    self.root,
                    version="2021_valid",
                    transform=self.transform_test,
                )
            else:
                self.ood = self.ood_dataset(
                    self.root,
                    transform=self.transform_test,
                    download=True,
                )

    def test_dataloader(self) -> list[DataLoader]:
        """Get the test dataloaders for ImageNet.

        Return:
            List[DataLoader]: ImageNet test set (in distribution data) and
            Textures test split (out-of-distribution data).
        """
        dataloader = [self._data_loader(self.test)]
        if self.evaluate_ood:
            dataloader.append(self._data_loader(self.ood))
        return dataloader

    @classmethod
    def add_argparse_args(
        cls,
        parent_parser: ArgumentParser,
        **kwargs: Any,
    ) -> ArgumentParser:
        p = super().add_argparse_args(parent_parser)

        # Arguments for ImageNet
        p.add_argument("--evaluate_ood", action="store_true")
        p.add_argument("--ood_ds", choices=cls.ood_datasets, default="svhn")
        p.add_argument("--test_alt", choices=cls.test_datasets, default=None)
        p.add_argument("--procedure", choices=["ViT", "A3"], default=None)
        p.add_argument("--train_size", type=int, default=224)
        p.add_argument(
            "--rand_augment", dest="rand_augment_opt", type=str, default=None
        )
        return parent_parser
