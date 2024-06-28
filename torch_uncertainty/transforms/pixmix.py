"""Code adapted from PixMix' paper."""

import numpy as np
from PIL import Image
from torch import nn

from torch_uncertainty.transforms import Shear, Translate, augmentations


def get_ab(beta: float) -> tuple[float, float]:
    rng = np.random.default_rng()
    if rng.uniform(low=0, high=1) < 0.5:
        a = np.float32(rng.beta(a=beta, b=1))
        b = np.float32(rng.beta(a=1, b=beta))
    else:
        a = 1 + np.float32(rng.beta(a=1, b=beta))
        b = -np.float32(rng.beta(a=1, b=beta))
    return a, b


def add(img1: Image.Image, img2: Image.Image, beta: float) -> Image.Image:
    a, b = get_ab(beta)
    img1, img2 = img1 * 2 - 1, img2 * 2 - 1
    out = a * img1 + b * img2
    return (out + 1) / 2


def multiply(img1: Image.Image, img2: Image.Image, beta: float) -> Image.Image:
    a, b = get_ab(beta)
    img1, img2 = img1 * 2, img2 * 2
    out = (img1**a) * (img2.clip(1e-37) ** b)
    return out / 2


# Summarize mixing operations
mixings = [add, multiply]


class PixMix(nn.Module):
    def __init__(
        self,
        mixing_set,
        mixing_iterations: int = 4,
        augmentation_severity: float = 3,
        mixing_severity: float = 3,
        all_ops: bool = True,
        seed: int = 12345,
    ) -> None:
        """PixMix augmentation class.

        Args:
            mixing_set (MixingSet): Dataset to be mixed with.
            mixing_iterations (int): Number of mixing iterations.
            augmentation_severity (float): Severity of augmentation.
            mixing_severity (float): Severity of mixing.
            all_ops (bool): Whether to use augmentations included in ImageNet-C.
                Defaults to True.
            seed (int): Seed for random number generator. Defaults to 12345.

        Note:
            Default arguments are set to follow original guidelines.
        """
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.mixing_set = mixing_set
        self.num_mixing_images = len(mixing_set)
        self.mixing_iterations = mixing_iterations
        self.augmentation_severity = augmentation_severity
        self.mixing_severity = mixing_severity

        if not all_ops:
            allowed_augmentations = [
                aug for aug in augmentations if not aug.corruption_overlap
            ]
        else:
            allowed_augmentations = augmentations

        self.aug_instances = []
        for aug in allowed_augmentations:
            if aug in (Shear, Translate):
                self.aug_instances.append(aug(axis=0))
                self.aug_instances.append(aug(axis=1))
            else:
                self.aug_instances.append(aug())

    def __call__(self, img: Image.Image) -> np.ndarray:
        # TODO: Fix
        mixed = self.augment_input(img) if self.rng.random() < 0.5 else img

        for _ in range(self.rng.integers(low=0, high=self.mixing_iterations)):
            if self.rng.random() < 0.5:
                aug_image_copy = self._augment(img)
            else:
                aug_image_copy = self.rng.choice(self.num_mixing_images)

            # TODO: Fix
            mixed_op = self.rng.choice(mixings)
            mixed = mixed_op(
                np.array(mixed), np.array(aug_image_copy), self.mixing_severity
            )
            mixed = np.clip(mixed, 0, 1)
        return mixed

    def _augment(self, image: Image.Image) -> np.ndarray:
        op = self.rng.choice(self.aug_instances, 1)
        if op.level_type is int:
            aug_level = self._sample_int(op.pixmix_max_level)
        else:
            aug_level = self._sample_float(op.pixmix_max_level)
        return op(image.copy(), aug_level)

    def _sample_level(self) -> float:
        return self.rng.uniform(low=0.1, high=self.augmentation_severity)

    def _sample_int(self, maxval: int) -> int:
        """Helper method to scale `level` between 0 and maxval.

        Args:
            level: Level of the operation that will be between [0, maxval]
            maxval: Maximum value that the operation can have. This will be
            scaled to level/maxval.

        Returns:
            An int that results from scaling `maxval` according to `level`.
        """
        return int(self._sample_level() * maxval / 10)

    def _sample_float(self, maxval: float) -> float:
        """Helper function to scale `val` between 0 and maxval.

        Args:
            level: Level of the operation that will be in the range
                [0, `maxval`]
            maxval: Maximum value that the operation can have. This will be
                scaled to level/maxval.

        Returns:
            A float that results from scaling `maxval` according to `level`.
        """
        return float(self._sample_level()) * maxval / 10.0
