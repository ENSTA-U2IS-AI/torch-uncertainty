import copy
from pathlib import Path

import torch
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torch.utils.data import random_split
from torchvision import datasets, tv_tensors
from torchvision.transforms import v2

from torch_uncertainty.datamodules.abstract import AbstractDataModule
from torch_uncertainty.datasets.segmentation import Cityscapes
from torch_uncertainty.transforms import RandomRescale


class CityscapesDataModule(AbstractDataModule):
    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        crop_size: _size_2_t = 1024,
        inference_size: _size_2_t = (1024, 2048),
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

        self.dataset = Cityscapes
        self.mode = "fine"
        self.crop_size = _pair(crop_size)
        self.inference_size = _pair(inference_size)

        self.train_transform = v2.Compose(
            [
                v2.ToImage(),
                RandomRescale(min_scale=0.5, max_scale=2.0, antialias=True),
                v2.RandomCrop(size=self.crop_size, pad_if_needed=True),
                v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                v2.RandomHorizontalFlip(),
                v2.ToDtype(
                    dtype={
                        tv_tensors.Image: torch.float32,
                        tv_tensors.Mask: torch.int64,
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
                v2.ToImage(),
                v2.Resize(size=self.inference_size, antialias=True),
                v2.ToDtype(
                    dtype={
                        tv_tensors.Image: torch.float32,
                        tv_tensors.Mask: torch.int64,
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
        self.dataset(root=self.root, split="train", mode=self.mode)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            full = datasets.wrap_dataset_for_transforms_v2(
                self.dataset(
                    root=self.root,
                    split="train",
                    mode=self.mode,
                    target_type="semantic",
                    transforms=self.train_transform,
                )
            )

            if self.val_split is not None:
                self.train, val = random_split(
                    full,
                    [
                        1 - self.val_split,
                        self.val_split,
                    ],
                )
                # FIXME: memory cost issues might arise here
                self.val = copy.deepcopy(val)
                self.val.dataset.transforms = self.test_transform
            else:
                self.train = full
                self.val = datasets.wrap_dataset_for_transforms_v2(
                    self.dataset(
                        root=self.root,
                        split="val",
                        mode=self.mode,
                        target_type="semantic",
                        transforms=self.test_transform,
                    )
                )

        if stage == "test" or stage is None:
            self.test = datasets.wrap_dataset_for_transforms_v2(
                self.dataset(
                    root=self.root,
                    split="val",
                    mode=self.mode,
                    target_type="semantic",
                    transforms=self.test_transform,
                )
            )

        if stage not in ["fit", "test", None]:
            raise ValueError(f"Stage {stage} is not supported.")
