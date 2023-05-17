# fmt: off
from PIL import Image

import numpy as np
from torch_uncertainty.datasets.mixing_set import MixingSet
from torch_uncertainty.transforms.generic import (
    augmentations,
    augmentations_all,
    mixings,
)


# fmt: on
class PixMix:
    """PixMix augmentation class.

    Args:
        mixing_set (MixingSet): Dataset to be mixed with.
        mixing_iterations (int): Number of mixing iterations.
        mixing_severity (float): Severity of mixing.
        all_ops (bool): Whether to use augmentations included in ImageNet-C.
            Defaults to True.
    """

    def __init__(
        self,
        mixing_set: MixingSet,
        mixing_iterations: int,
        mixing_severity: float,
        all_ops: bool = True,
    ):
        self.mixing_set = mixing_set
        self.num_mixing_images = len(mixing_set)
        self.mixing_iterations = mixing_iterations
        self.mixing_severity = mixing_severity
        self.augmentations = augmentations_all if all_ops else augmentations

    def __call__(self, img: Image) -> np.ndarray:
        if np.random.random() < 0.5:
            mixed = self.augment_input(img)
        else:
            mixed = img

        for _ in range(np.random.randint(self.mixing_iterations + 1)):
            if np.random.random() < 0.5:
                aug_image_copy = self.augment_input(img)
            else:
                aug_image_copy = np.random.choice(self.num_mixing_images)

            mixed_op = np.random.choice(mixings)
            mixed = mixed_op(
                np.array(mixed), np.array(aug_image_copy), self.mixing_severity
            )
            mixed = np.clip(mixed, 0, 1)
        return mixed

    def augment_input(self, image: Image, aug_severity: int = 1) -> np.ndarray:
        op = np.random.choice(self.augmentations)
        return op(image.copy(), aug_severity)
