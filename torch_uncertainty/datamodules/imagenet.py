# fmt: off
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional, Union

import torch.nn as nn
import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from timm.data.auto_augment import rand_augment_transform
from timm.data.mixup import Mixup
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import DTD, SVHN, ImageNet, INaturalist

from ..datasets import ImageNetA, ImageNetO, ImageNetR


# fmt: on
class ImageNetDataModule(LightningDataModule):
    num_classes = 1000
    num_channels = 3
    test_datasets = ["r", "o", "a"]
    ood_datasets = ["inaturalist", "imagenet-o", "svhn", "textures"]
    training_task = "classification"

    def __init__(
        self,
        root: Union[str, Path],
        ood_detection: bool,
        batch_size: int,
        ood_ds: str = "svhn",
        test_alt: Optional[str] = None,
        procedure: str = "A3",
        train_size: int = 224,
        rand_augment_opt: Optional[str] = None,
        num_workers: int = 1,
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
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
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
            train_size = train_size
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

    def prepare_data(self) -> None:
        if self.test_alt is not None:
            self.data = self.dataset(
                self.root,
                split="val",
                download=True,
            )
        if self.ood_detection:
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

    def setup(self, stage: Optional[str] = None) -> None:
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
        if stage == "test":
            self.test = self.dataset(
                self.root,
                split="val",
                transform=self.transform_test,
            )

        if self.ood_detection:
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

    def train_dataloader(self) -> DataLoader:
        """Get the training dataloader for ImageNet.

        Return:
            DataLoader: ImageNet training dataloader.
        """
        return self._data_loader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Get the validation dataloader for ImageNet.

        Return:
            DataLoader: ImageNet validation dataloader.
        """
        return self._data_loader(self.val)

    def test_dataloader(self) -> List[DataLoader]:
        """Get the test dataloaders for ImageNet.

        Return:
            List[DataLoader]: ImageNet test set (in distribution data) and
            Textures test split (out-of-distribution data).
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
        p.add_argument("--batch_size", type=int, default=256)
        p.add_argument("--num_workers", type=int, default=4)
        p.add_argument(
            "--evaluate_ood", dest="ood_detection", action="store_true"
        )
        p.add_argument("--ood_ds", choices=cls.ood_datasets, default="svhn")
        p.add_argument("--test_alt", choices=cls.test_datasets, default=None)
        p.add_argument("--procedure", choices=["ViT", "A3"], default=None)
        p.add_argument("--train_size", type=int, default=224)
        p.add_argument(
            "--rand_augment", dest="rand_augment_opt", type=str, default=None
        )
        return parent_parser
