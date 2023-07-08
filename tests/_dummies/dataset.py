# fmt:off
from typing import Any, Callable, Tuple

import torch
import torch.utils.data as data
from PIL import Image

import numpy as np


# fmt:on
class DummyDataset(data.Dataset):
    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable[..., Any] | None = None,
        target_transform: Callable[..., Any] | None = None,
        num_channels: int = 3,
        image_size: int = 4,
        num_classes: int = 10,
        num_images: int = 2,
        **kwargs: Any,
    ) -> None:
        """Dummy dataset for testing purposes.

        Args:
            root (string): Root directory containing the dataset (unused).
            train (bool, optional): If True, creates dataset from training set,
                otherwise creates from test set (unused).
            transform (callable, optional): A function/transform that takes in
                a PIL image and returns a transformed version. E.g,
                ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that
                takes in the target and transforms it.
            num_channels (int, optional): Number of channels in the images.
            image_size (int, optional): Size of the images.
            num_classes (int, optional): Number of classes in the dataset.
            num_images (int, optional): Number of images in the dataset.
        """
        self.root = root
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform

        self.data: Any = []
        self.targets = []

        if num_channels == 1:
            shape = (num_images, image_size, image_size)
        else:
            shape = (num_images, num_channels, image_size, image_size)

        self.data = np.random.randint(
            low=0,
            high=255,
            size=shape,
            dtype=np.uint8,
        )

        if num_channels == 1:
            self.data = self.data.transpose((0, 1, 2))  # convert to HWC
        else:
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.targets = torch.randint(
            low=0, high=num_classes, size=(num_images,)
        )
        self.targets = torch.arange(start=0, end=num_classes).repeat(
            num_images // (num_classes) + 1
        )[:num_images]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
