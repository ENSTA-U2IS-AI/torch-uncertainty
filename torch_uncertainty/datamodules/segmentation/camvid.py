from pathlib import Path

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from torch_uncertainty.datamodules.abstract import AbstractDataModule
from torch_uncertainty.datasets.segmentation import CamVid


class CamVidDataModule(AbstractDataModule):
    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        val_split: float | None = None,  # FIXME: not used for now
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
        self.dataset = CamVid

        self.train_transform = v2.Compose(
            [
                v2.Resize(
                    (360, 480), interpolation=v2.InterpolationMode.NEAREST
                ),
                v2.ToDtype(
                    dtype={
                        tv_tensors.Image: torch.float32,
                        tv_tensors.Mask: torch.int64,
                        "others": None,
                    },
                    scale=True,
                ),
            ]
        )
        self.test_transform = v2.Compose(
            [
                v2.Resize(
                    (360, 480), interpolation=v2.InterpolationMode.NEAREST
                ),
                v2.ToDtype(
                    dtype={
                        tv_tensors.Image: torch.float32,
                        tv_tensors.Mask: torch.int64,
                        "others": None,
                    },
                    scale=True,
                ),
            ]
        )

    def prepare_data(self) -> None:  # coverage: ignore
        self.dataset(root=self.root, download=True)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train = self.dataset(
                root=self.root,
                split="train",
                download=False,
                transforms=self.train_transform,
            )
            self.val = self.dataset(
                root=self.root,
                split="val",
                download=False,
                transforms=self.test_transform,
            )
        if stage == "test" or stage is None:
            self.test = self.dataset(
                root=self.root,
                split="test",
                download=False,
                transforms=self.test_transform,
            )

        if stage not in ["fit", "test", None]:
            raise ValueError(f"Stage {stage} is not supported.")
