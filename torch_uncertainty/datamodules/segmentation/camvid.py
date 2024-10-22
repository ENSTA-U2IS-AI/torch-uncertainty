import logging
from pathlib import Path

import torch
from torch import nn
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torchvision import tv_tensors
from torchvision.transforms import v2

from torch_uncertainty.datamodules import TUDataModule
from torch_uncertainty.datasets.segmentation import CamVid
from torch_uncertainty.transforms import RandomRescale


class CamVidDataModule(TUDataModule):
    num_channels = 3
    training_task = "segmentation"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        crop_size: _size_2_t = 640,
        eval_size: _size_2_t = (720, 960),
        group_classes: bool = True,
        basic_augment: bool = True,
        val_split: float | None = None,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        r"""DataModule for the CamVid dataset.

        Args:
            root (str or Path): Root directory of the datasets.
            batch_size (int): Number of samples per batch.
            crop_size (sequence or int, optional): Desired input image and
                segmentation mask sizes during training. If :attr:`crop_size` is an
                int instead of sequence like :math:`(H, W)`, a square crop
                :math:`(\text{size},\text{size})` is made. If provided a sequence
                of length :math:`1`, it will be interpreted as
                :math:`(\text{size[0]},\text{size[1]})`. Defaults to ``640``.
            eval_size (sequence or int, optional): Desired input image and
                segmentation mask sizes during evaluation. If size is an int,
                smaller edge of the images will be matched to this number, i.e.,
                :math:`\text{height}>\text{width}`, then image will be rescaled to
                :math:`(\text{size}\times\text{height}/\text{width},\text{size})`.
                Defaults to ``(720,960)``.
            group_classes (bool, optional): Whether to group the 32 classes into
                11 superclasses. Default: ``True``.
            basic_augment (bool): Whether to apply base augmentations. Defaults to
                ``True``.
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
                        v2.Resize(640),
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
        if val_split is not None:  # coverage: ignore
            logging.warning("val_split is not used for CamVidDataModule.")

        super().__init__(
            root=root,
            batch_size=batch_size,
            val_split=val_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        if group_classes:
            self.num_classes = 11
        else:
            self.num_classes = 32
        self.dataset = CamVid
        self.group_classes = group_classes
        self.crop_size = _pair(crop_size)
        self.eval_size = _pair(eval_size)

        if basic_augment:
            basic_transform = v2.Compose(
                [
                    RandomRescale(min_scale=0.5, max_scale=2.0, antialias=True),
                    v2.RandomCrop(
                        size=self.crop_size,
                        pad_if_needed=True,
                        fill={tv_tensors.Image: 0, tv_tensors.Mask: 255},
                    ),
                    v2.ColorJitter(
                        brightness=0.5, contrast=0.5, saturation=0.5
                    ),
                    v2.RandomHorizontalFlip(),
                ]
            )
        else:
            basic_transform = nn.Identity()

        self.train_transform = v2.Compose(
            [
                v2.ToImage(),
                basic_transform,
                v2.ToDtype(
                    dtype={
                        tv_tensors.Image: torch.float32,
                        tv_tensors.Mask: torch.int64,
                        "others": None,
                    },
                    scale=True,
                ),
                v2.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self.test_transform = v2.Compose(
            [
                v2.ToImage(),
                v2.Resize(size=self.eval_size, antialias=True),
                v2.ToDtype(
                    dtype={
                        tv_tensors.Image: torch.float32,
                        tv_tensors.Mask: torch.int64,
                        "others": None,
                    },
                    scale=True,
                ),
                v2.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def prepare_data(self) -> None:  # coverage: ignore
        self.dataset(root=self.root, download=True)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train = self.dataset(
                root=self.root,
                split="train",
                group_classes=self.group_classes,
                download=False,
                transforms=self.train_transform,
            )
            self.val = self.dataset(
                root=self.root,
                split="val",
                group_classes=self.group_classes,
                download=False,
                transforms=self.test_transform,
            )
        if stage == "test" or stage is None:
            self.test = self.dataset(
                root=self.root,
                split="test",
                group_classes=self.group_classes,
                download=False,
                transforms=self.test_transform,
            )

        if stage not in ["fit", "test", None]:
            raise ValueError(f"Stage {stage} is not supported.")
