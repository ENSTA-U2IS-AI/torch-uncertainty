from pathlib import Path

import torch
from torchvision import tv_tensors
from torchvision.transforms import v2

from torch_uncertainty.datamodules.abstract import TUDataModule
from torch_uncertainty.datasets.segmentation import CamVid


class CamVidDataModule(TUDataModule):
    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        val_split: float | None = None,  # FIXME: not used for now
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        r"""DataModule for the CamVid dataset.

        Args:
            root (str or Path): Root directory of the datasets.
            batch_size (int): Number of samples per batch.
            val_split (float or None, optional): Share of training samples to use
                for validation. Defaults to ``None``.
            num_workers (int, optional): Number of dataloaders to use. Defaults to
                ``1``.
            pin_memory (bool, optional):  Whether to pin memory. Defaults to
                ``True``.
            persistent_workers (bool, optional): Whether to use persistent workers.
                Defaults to ``True``.

        Note:
            This datamodule injects the following transforms into the training and
            validation/test datasets:

            .. code-block:: python

                from torchvision.transforms import v2

                v2.Compose(
                    [
                        v2.Resize((360, 480)),
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

            This behavior can be modified by overriding ``self.train_transform``
            and ``self.test_transform`` after initialization.
        """
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
                v2.Resize((360, 480)),
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
                v2.Resize((360, 480)),
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
