from pathlib import Path
from typing import Literal

import torch
from torch import nn
from torch.nn.common_types import _size_2_t
from torch.nn.modules.utils import _pair
from torchvision import tv_tensors
from torchvision.transforms import v2

from torch_uncertainty.datamodules import TUDataModule
from torch_uncertainty.datasets import MUAD
from torch_uncertainty.datasets.utils import create_train_val_split
from torch_uncertainty.transforms import RandomRescale


class MUADDataModule(TUDataModule):
    num_classes = 15
    num_channels = 3
    training_task = "segmentation"
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        version: Literal["full", "small"] = "full",
        eval_batch_size: int | None = None,
        eval_ood: bool = False,
        crop_size: _size_2_t = 1024,
        eval_size: _size_2_t = (1024, 2048),
        train_transform: nn.Module | None = None,
        test_transform: nn.Module | None = None,
        val_split: float | None = None,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        r"""Segmentation DataModule for the MUAD dataset.

        Args:
            root (str or Path): Root directory of the datasets.
            batch_size (int): Number of samples per batch during training.
            version (str, optional): Version of the dataset to use. Can be either
                ``full`` or ``small``. Defaults to ``full``.
            eval_batch_size (int | None) : Number of samples per batch during evaluation (val
                and test). Set to :attr:`batch_size` if ``None``. Defaults to ``None``.
            eval_ood (bool): Whether to evaluate on the OOD dataset. Defaults to
                ``False``. If set to ``True``, the OOD dataset will be used for
                evaluation in addition of the test dataset.
            crop_size (sequence or int, optional): Desired input image and
                segmentation mask sizes during training. If :attr:`crop_size` is an
                int instead of sequence like :math:`(H, W)`, a square crop
                :math:`(\text{size},\text{size})` is made. If provided a sequence
                of length :math:`1`, it will be interpreted as
                :math:`(\text{size[0]},\text{size[1]})`. Has to be provided if
                :attr:`train_transform` is not provided. Otherwise has no effect.
                Defaults to ``1024``.
            eval_size (sequence or int, optional): Desired input image and
                segmentation mask sizes during inference. If size is an int,
                smaller edge of the images will be matched to this number, i.e.,
                :math:`\text{height}>\text{width}`, then image will be rescaled to
                :math:`(\text{size}\times\text{height}/\text{width},\text{size})`.
                Has to be provided if :attr:`test_transform` is not provided.
                Otherwise has no effect. Defaults to ``(1024,2048)``.
            train_transform (nn.Module | None): Custom training transform. Defaults
                to ``None``. If not provided, a default transform is used.
            test_transform (nn.Module | None): Custom test transform. Defaults to
                ``None``. If not provided, a default transform is used.
            val_split (float or None, optional): Share of training samples to use
                for validation. Defaults to ``None``.
            num_workers (int, optional): Number of dataloaders to use. Defaults to
                ``1``.
            pin_memory (bool, optional): Whether to pin memory. Defaults to
                ``True``.
            persistent_workers (bool, optional): Whether to use persistent workers.
                Defaults to ``True``.

        Note:
            By default this datamodule injects the following transforms into the training and
            validation/test datasets:

            Training transforms:

            .. code-block:: python

                from torchvision.transforms import v2

                v2.Compose(
                    [
                        RandomRescale(min_scale=0.5, max_scale=2.0, antialias=True),
                        v2.RandomCrop(size=crop_size, pad_if_needed=True),
                        v2.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                        v2.RandomHorizontalFlip(),
                        v2.ToDtype(
                            {
                                tv_tensors.Image: torch.float32,
                                tv_tensors.Mask: torch.int64,
                                "others": None,
                            },
                            scale=True,
                        ),
                        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                )

            Validation/Test transforms:

            .. code-block:: python

                from torchvision.transforms import v2

                v2.Compose(
                    [
                        v2.Resize(size=eval_size, antialias=True),
                        v2.ToDtype(
                            {
                                tv_tensors.Image: torch.float32,
                                tv_tensors.Mask: torch.int64,
                                "others": None,
                            },
                            scale=True,
                        ),
                        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    ]
                )

            This behavior can be modified by setting up ``train_transform``
            and ``test_transform`` at initialization.
        """
        super().__init__(
            root=root,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            val_split=val_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.dataset = MUAD
        self.version = version
        self.eval_ood = eval_ood
        self.crop_size = _pair(crop_size)
        self.eval_size = _pair(eval_size)

        # FIXME: should be the same split names (update huggingface dataset)
        self.test_split = "test" if version == "small" else "test_id"
        self.ood_split = "ood" if version == "small" else "test_ood"

        if train_transform is not None:
            self.train_transform = train_transform
        else:
            if version == "small":
                self.train_transform = v2.Compose(
                    [
                        v2.Resize(size=self.eval_size, antialias=True),
                        v2.RandomHorizontalFlip(),
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
            else:
                self.train_transform = v2.Compose(
                    [
                        v2.Resize(size=self.eval_size, antialias=True),
                        RandomRescale(min_scale=0.5, max_scale=2.0, antialias=True),
                        v2.RandomCrop(
                            size=self.crop_size,
                            pad_if_needed=True,
                            fill={tv_tensors.Image: 0, tv_tensors.Mask: 255},
                        ),
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
                        v2.Normalize(mean=self.mean, std=self.std),
                    ]
                )

        if test_transform is not None:
            self.test_transform = test_transform
        else:
            self.test_transform = v2.Compose(
                [
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
        self.dataset(
            root=self.root,
            split="train",
            version=self.version,
            target_type="semantic",
            download=True,
        )
        self.dataset(
            root=self.root, split="val", version=self.version, target_type="semantic", download=True
        )
        self.dataset(
            root=self.root,
            split=self.test_split,
            version=self.version,
            target_type="semantic",
            download=True,
        )

        if self.eval_ood:
            self.dataset(
                root=self.root,
                split=self.ood_split,
                version=self.version,
                target_type="semantic",
                download=True,
            )

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            full = self.dataset(
                root=self.root,
                split="train",
                version=self.version,
                target_type="semantic",
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
                    split="val",
                    version=self.version,
                    target_type="semantic",
                    transforms=self.test_transform,
                )

        if stage == "test" or stage is None:
            self.test = self.dataset(
                root=self.root,
                split=self.test_split,
                version=self.version,
                target_type="semantic",
                transforms=self.test_transform,
            )
            if self.eval_ood:
                self.ood = self.dataset(
                    root=self.root,
                    split=self.ood_split,
                    version=self.version,
                    target_type="semantic",
                    transforms=self.test_transform,
                )

        if stage not in ["fit", "test", None]:
            raise ValueError(f"Stage {stage} is not supported.")

    def test_dataloader(self) -> torch.utils.data.DataLoader:
        """Returns the test dataloader."""
        dataloader = [self._data_loader(self.get_test_set(), training=False, shuffle=False)]
        if self.eval_ood:
            dataloader.append(self._data_loader(self.get_ood_set(), training=False, shuffle=False))
        return dataloader
