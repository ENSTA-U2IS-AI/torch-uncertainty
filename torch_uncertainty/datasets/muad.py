import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    download_url,
)


class MUAD(VisionDataset):
    classes_url = "https://raw.githubusercontent.com/torch-uncertainty/dataset-metadata/main/segmentation/muad/classes.json"
    classes_md5 = "1db6e6143939824792f0af11a4fe7bb1"  # avoid replacement attack
    base_url = "https://zenodo.org/records/10619959/files/"

    zip_md5 = {
        "train": "cea6a672225b10dda1add8b2974a5982",
        "train_depth": "934d122ac09e0471db62ae68c3456b0f",
        "val": "957af9c1c36f0a85c33279e06b6cf8d8",
        "val_depth": "0282030d281aeffee3335f713ba12373",
    }
    samples: list[Path] = []
    targets: list[Path] = []

    # TODO: Add depth regression mode
    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "train_depth", "val_depth"],
        transform: Callable | None = None,
        download: bool = False,
    ) -> None:
        """The MUAD Dataset.

        Args:
            root (str): Root directory of dataset where directory 'leftImg8bit'
                and 'leftLabel' are located.
            split (str, optional): The image split to use, 'train', 'val',
            'train_depth' or 'val_depth'.
            transform (callable, optional): A function/transform that takes in
                a tuple of PIL images and returns a transformed version.
            download (bool, optional): If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.

        Reference:
            https://muad-dataset.github.io

        Note:
            MUAD cannot be used for commercial purposes. Read MUAD's license
            carefully before using it and verify that you can comply.
        """
        print(
            "MUAD is restricted to non-commercial use. By using MUAD, you "
            "agree to the terms and conditions."
        )
        super().__init__(
            root=Path(root) / "MUAD",
            transform=transform,
        )

        if split not in ["train", "val", "train_depth", "val_depth"]:
            raise ValueError(
                "split must be one of ['train', 'val', 'train_depth', "
                f"'val_depth']. Got {split}."
            )
        self.split = split

        split_path = self.root / (split + ".zip")
        if not check_integrity(split_path, self.zip_md5[split]) and download:
            self._download()

        # Load classes metadata
        cls_path = self.root / "classes.json"
        if not check_integrity(cls_path, self.classes_md5) and download:
            download_url(
                self.classes_url,
                self.root,
                "classes.json",
                self.classes_md5,
            )

        with (self.root / "classes.json").open() as file:
            self.classes = json.load(file)

        train_id_to_color = [
            c["object_id"]
            for c in self.classes
            if c["train_id"] not in [-1, 255]
        ]
        train_id_to_color.append([0, 0, 0])
        self.train_id_to_color = np.array(train_id_to_color)

        self._make_dataset(self.root / split)

    def decode_target(self, target: Image.Image) -> np.ndarray:
        target[target == 255] = 19
        return self.train_id_to_color[target]

    def __getitem__(self, index: int) -> tuple[Image.Image, Image.Image]:
        """Get the image and its segmentation target."""
        img_path = self.samples[index]
        seg_path = self.targets[index]

        image = cv.imread(img_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        segm = plt.imread(seg_path) * 255.0
        target = np.zeros((segm.shape[0], segm.shape[1])) + 255.0

        for c in self.classes:
            upper = np.array(c["train_id"])
            mask = cv.inRange(segm, upper, upper)
            target[mask == 255] = c["train_id"]
        target = target.astype(np.uint8)
        target = Image.fromarray(target)

        image = Image.fromarray(image)

        if self.transform:
            image, target = self.transform(image, target)
        return image, target

    def __len__(self) -> int:
        """The number of samples in the dataset."""
        return len(self.samples)

    def _make_dataset(self, path: Path) -> None:
        """Create a list of samples and targets.

        Args:
            path (Path): The path to the dataset.
        """
        if "depth" in path.name:
            raise NotImplementedError(
                "Depth regression mode is not implemented yet. Raise an issue "
                "if you need it."
            )
        self.samples = list((path / "leftImg8bit/").glob("**/*"))
        self.targets = list((path / "leftLabel/").glob("**/*"))

    def _download(self):
        """Download and extract the chosen split of the dataset."""
        split_url = self.base_url + self.split + ".zip"
        download_and_extract_archive(
            split_url, self.root, md5=self.zip_md5[self.split]
        )
