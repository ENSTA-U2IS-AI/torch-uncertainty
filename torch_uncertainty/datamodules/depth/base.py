from pathlib import Path

import torch
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torchvision import tv_tensors
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2

from torch_uncertainty.datamodules.abstract import AbstractDataModule
from torch_uncertainty.transforms import RandomRescale
from torch_uncertainty.utils.misc import create_train_val_split


class DepthDataModule(AbstractDataModule):
    def __init__(
        self,
        dataset: type[VisionDataset],
        root: str | Path,
        batch_size: int,
        max_depth: float,
        crop_size: _size_2_t,
        inference_size: _size_2_t,
        val_split: float | None = None,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__(
            root=root,
            batch_size=batch_size,
            val_split=val_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.dataset = dataset
        self.max_depth = max_depth
        self.crop_size = _pair(crop_size)
        self.inference_size = _pair(inference_size)

        self.train_transform = v2.Compose(
            [
                RandomRescale(min_scale=0.5, max_scale=2.0, antialias=True),
                v2.RandomCrop(
                    size=self.crop_size,
                    pad_if_needed=True,
                    fill={tv_tensors.Image: 0, tv_tensors.Mask: float("nan")},
                ),
                v2.RandomHorizontalFlip(),
                v2.ToDtype(
                    dtype={
                        tv_tensors.Image: torch.float32,
                        "others": None,
                    },
                    scale=True,
                ),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.test_transform = v2.Compose(
            [
                v2.Resize(size=self.inference_size, antialias=True),
                v2.ToDtype(
                    dtype={
                        tv_tensors.Image: torch.float32,
                        "others": None,
                    },
                    scale=True,
                ),
                v2.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def prepare_data(self) -> None:  # coverage: ignore
        self.dataset(
            root=self.root,
            split="train",
            max_depth=self.max_depth,
            download=True,
        )
        self.dataset(
            root=self.root, split="val", max_depth=self.max_depth, download=True
        )

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            full = self.dataset(
                root=self.root,
                max_depth=self.max_depth,
                split="train",
                transforms=self.train_transform,
            )

            if self.val_split is not None:
                self.train, self.val = create_train_val_split(
                    full,
                    self.val_split,
                    self.test_transform,
                )
            else:
                self.train = full
                self.val = self.dataset(
                    root=self.root,
                    max_depth=self.max_depth,
                    split="val",
                    transforms=self.test_transform,
                )

        if stage == "test" or stage is None:
            self.test = self.dataset(
                root=self.root,
                max_depth=self.max_depth,
                split="val",
                transforms=self.test_transform,
            )

        if stage not in ["fit", "test", None]:
            raise ValueError(f"Stage {stage} is not supported.")
