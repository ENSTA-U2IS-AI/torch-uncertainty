from pathlib import Path

from torch import nn
from torch.nn.common_types import _size_2_t

from torch_uncertainty.datasets import NYUv2

from .base import DepthDataModule


class NYUv2DataModule(DepthDataModule):
    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        eval_batch_size: int | None = None,
        min_depth: float = 1e-3,
        max_depth: float = 10.0,
        crop_size: _size_2_t = (416, 544),
        eval_size: _size_2_t = (480, 640),
        train_transform: nn.Module | None = None,
        test_transform: nn.Module | None = None,
        val_split: float | None = None,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        r"""Depth DataModule for the NYUv2 dataset.

        Args:
            root (str or Path): Root directory of the datasets.
            batch_size (int): Number of samples per batch during training.
            eval_batch_size (int | None) : Number of samples per batch during evaluation (val
                and test). Set to :attr:`batch_size` if ``None``. Defaults to ``None``.
            min_depth (float, optional): Minimum depth value for evaluation.
                Defaults to ``1e-3``.
            max_depth (float, optional): Maximum depth value for training and
                evaluation. Defaults to ``10.0``.
            crop_size (sequence or int, optional): Desired input image and
                depth mask sizes during training. If :attr:`crop_size` is an
                int instead of sequence like :math:`(H, W)`, a square crop
                :math:`(\text{size},\text{size})` is made. If provided a sequence
                of length :math:`1`, it will be interpreted as
                :math:`(\text{size[0]},\text{size[1]})`. Has to be provided if
                :attr:`train_transform` is not provided. Otherwise has no effect.
                Defaults to ``(416, 544)``.
            eval_size (sequence or int, optional): Desired input image and
                depth mask sizes during evaluation. If size is an int,
                smaller edge of the images will be matched to this number, i.e.,
                :math:`\text{height}>\text{width}`, then image will be rescaled to
                :math:`(\text{size}\times\text{height}/\text{width},\text{size})`.
                Has to be provided if :attr:`test_transform` is not provided.
                Otherwise has no effect. Defaults to ``(480, 640)``.
            train_transform (nn.Module | None): Custom training transform. Defaults
                to ``None``. If not provided, a default transform is used.
            test_transform (nn.Module | None): Custom test transform. Defaults to
                ``None``. If not provided, a default transform is used.
            val_split (float or None, optional): Share of training samples to use
                for validation. Defaults to ``None``.
            num_workers (int, optional): Number of dataloaders to use. Defaults to
                ``1``.
            pin_memory (bool, optional):  Whether to pin memory. Defaults to
                ``True``.
            persistent_workers (bool, optional): Whether to use persistent workers.
                Defaults to ``True``.
        """
        super().__init__(
            dataset=NYUv2,
            root=root,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            min_depth=min_depth,
            max_depth=max_depth,
            crop_size=crop_size,
            eval_size=eval_size,
            train_transform=train_transform,
            test_transform=test_transform,
            val_split=val_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
