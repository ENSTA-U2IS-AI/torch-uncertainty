# fmt: off
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, List, Optional, Union

import torchvision.transforms as T
from pytorch_lightning import LightningDataModule
from timm.data.auto_augment import rand_augment_transform
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.datasets import DTD, SVHN, ImageNet, INaturalist

from ..datasets import ImageNetO, ImageNetR


# fmt: on
class ImageNetDataModule(LightningDataModule):
    def __init__(
        self,
        root: Union[str, Path],
        batch_size: int,
        val_split: int = 0,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        ood_name: str = "svhn",
        test_option: str = None,
        # num_ops: int = 2,
        # magnitude: float = 15,
        procedure: str = "A3",
        **kwargs,
    ) -> None:
        super().__init__()

        if isinstance(root, str):
            root = Path(root)

        self.root: Path = root
        self.batch_size = batch_size
        self.val_split = val_split
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.ood_name = ood_name
        self.num_classes = 1000
        self.num_channels = 3

        if test_option is None:
            self.dataset = ImageNet
        elif test_option == "imagenet-r":
            self.dataset = ImageNetR
        else:
            raise ValueError(f"Error, {test_option} not taken in charge.")

        if ood_name == "inaturalist":
            self.ood_dataset = INaturalist
        elif ood_name == "imagenet-o":
            self.ood_dataset = ImageNetO
        elif ood_name == "svhn":
            self.ood_dataset = SVHN
        elif ood_name == "textures":
            self.ood_dataset = DTD
        else:
            raise ValueError(f"The dataset {ood_name} is not supported.")

        self.procedure = procedure

        if self.procedure == "Classic":
            print("Classic Procedure")
            train_size = 224
        elif self.procedure == "A3":
            print("Procedure A3")
            train_size = 160
        else:
            raise ValueError("The procedure is unknown")

        self.transform_train = T.Compose(
            [
                T.RandomResizedCrop(train_size),
                T.RandomHorizontalFlip(),
                rand_augment_transform("rand-m6-mstd0.5-inc1", {}),
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
        if self.ood_name == "inaturalist":
            self.ood = self.ood_dataset(
                self.root,
                version="2021_valid",
                download=True,
                transform=self.transform_test,
            )
        elif self.ood_name != "textures":
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
            full = self.dataset(
                self.root,
                split="train",
                transform=self.transform_train,
            )
            self.train, self.val = random_split(
                full, [len(full) - self.val_split, self.val_split]
            )
            if self.val_split == 0:
                self.val = self.dataset(
                    self.root,
                    split="val",
                    transform=self.transform_test,
                )
        if stage == "test" or stage is None:
            self.test = self.dataset(
                self.root,
                split="val",
                transform=self.transform_test,
            )
            if self.ood_name == "inaturalist":
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
        return self._data_loader(self.train, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._data_loader(self.val)

    def test_dataloader(self) -> List[DataLoader]:
        r"""Gets test dataloaders for ImageNet.
        Returns:
            List[DataLoader]: ImageNet test set (in distribution data) and SVHN
                test split (out-of-distribution data).
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
        p.add_argument("--batch_size", type=int, default=256)
        p.add_argument("--val_split", type=int, default=0)
        p.add_argument("--num_workers", type=int, default=4)
        p.add_argument("--ood_name", type=str, default="svhn")
        p.add_argument("--test_option", type=str)
        return parent_parser
