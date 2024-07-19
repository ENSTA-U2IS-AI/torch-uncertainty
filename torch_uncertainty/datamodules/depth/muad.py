from pathlib import Path

from torch.nn.common_types import _size_2_t

from torch_uncertainty.datasets import MUAD
from torch_uncertainty.utils.misc import create_train_val_split

from .base import DepthDataModule


class MUADDataModule(DepthDataModule):
    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        min_depth: float,
        max_depth: float,
        crop_size: _size_2_t = 1024,
        eval_size: _size_2_t = (1024, 2048),
        val_split: float | None = None,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        r"""Depth DataModule for the MUAD dataset.

        Args:
            root (str or Path): Root directory of the datasets.
            batch_size (int): Number of samples per batch.
            min_depth (float, optional): Minimum depth value for evaluation
            max_depth (float, optional): Maximum depth value for training and
                evaluation.
            crop_size (sequence or int, optional): Desired input image and
                depth mask sizes during training. If :attr:`crop_size` is an
                int instead of sequence like :math:`(H, W)`, a square crop
                :math:`(\text{size},\text{size})` is made. If provided a sequence
                of length :math:`1`, it will be interpreted as
                :math:`(\text{size[0]},\text{size[1]})`. Defaults to ``1024``.
            eval_size (sequence or int, optional): Desired input image and
                depth mask sizes during evaluation. If size is an int,
                smaller edge of the images will be matched to this number, i.e.,
                :math:`\text{height}>\text{width}`, then image will be rescaled to
                :math:`(\text{size}\times\text{height}/\text{width},\text{size})`.
                Defaults to ``(1024,2048)``.
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
            dataset=MUAD,
            root=root,
            batch_size=batch_size,
            min_depth=min_depth,
            max_depth=max_depth,
            crop_size=crop_size,
            eval_size=eval_size,
            val_split=val_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

    def prepare_data(self) -> None:  # coverage: ignore
        self.dataset(
            root=self.root,
            split="train",
            max_depth=self.max_depth,
            target_type="depth",
            download=True,
        )
        self.dataset(
            root=self.root,
            split="val",
            min_depth=self.min_depth,
            max_depth=self.max_depth,
            target_type="depth",
            download=True,
        )

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            full = self.dataset(
                root=self.root,
                split="train",
                max_depth=self.max_depth,
                target_type="depth",
                transforms=self.train_transform,
            )

            if self.val_split is not None:
                self.train, self.val = create_train_val_split(
                    full,
                    self.val_split,
                    self.test_transform,
                )
                self.val.min_depth = self.min_depth
            else:
                self.train = full
                self.val = self.dataset(
                    root=self.root,
                    split="val",
                    min_depth=self.min_depth,
                    max_depth=self.max_depth,
                    target_type="depth",
                    transforms=self.test_transform,
                )

        if stage == "test" or stage is None:
            self.test = self.dataset(
                root=self.root,
                split="val",
                min_depth=self.min_depth,
                max_depth=self.max_depth,
                target_type="depth",
                transforms=self.test_transform,
            )

        if stage not in ["fit", "test", None]:
            raise ValueError(f"Stage {stage} is not supported.")
