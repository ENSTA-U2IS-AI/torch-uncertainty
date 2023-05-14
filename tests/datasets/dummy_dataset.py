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
        download: bool = False,
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
            download (bool, optional): If True, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again (unused).
        """
        self.root = root
        self.train = train  # training set or test set
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data: Any = []
        self.targets = []

        self.data = np.random.randint(low=0, high=255, size=(10, 3, 32, 32))
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self.targets = torch.randint(low=0, high=10, size=(10,))

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

    def extra_repr(self) -> str:
        split = "Train" if self.train is True else "Test"
        return f"Split: {split}"
