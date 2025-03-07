import logging
import os
import shutil
from collections.abc import Callable
from importlib import util
from pathlib import Path
from typing import Literal, NamedTuple

from huggingface_hub import hf_hub_download
from PIL import Image

if util.find_spec("cv2"):
    import cv2

    cv2_installed = True
else:  # coverage: ignore
    cv2_installed = False
import numpy as np
from torchvision import tv_tensors
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    download_and_extract_archive,
)


class MUADClass(NamedTuple):
    name: str
    id: int
    color: tuple[int, int, int]


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

    small_muad_url = "ENSTA-U2IS/miniMUAD"

    _num_samples = {
        "full": {
            "train": 3420,
            "val": 492,
            "test": ...,
        },
        "small": {
            "train": 400,
            "val": 54,
            "test": 112,
            "ood": 20,
        },
    }

    classes = [
        MUADClass("road", 0, (128, 64, 128)),
        MUADClass("sidewalk", 1, (244, 35, 232)),
        MUADClass("building", 2, (70, 70, 70)),
        MUADClass("wall", 3, (102, 102, 156)),
        MUADClass("fence", 4, (190, 153, 153)),
        MUADClass("pole", 5, (153, 153, 153)),
        MUADClass("traffic_light", 6, (250, 170, 30)),
        MUADClass("traffic_sign", 7, (220, 220, 0)),
        MUADClass("vegetation", 8, (107, 142, 35)),
        MUADClass("terrain", 9, (152, 251, 152)),
        MUADClass("sky", 10, (70, 130, 180)),
        MUADClass("person", 11, (220, 20, 60)),
        MUADClass("rider", 12, (255, 0, 0)),
        MUADClass("car", 13, (0, 0, 142)),
        MUADClass("truck", 14, (0, 0, 70)),
        MUADClass("bus", 15, (0, 60, 100)),
        MUADClass("train", 16, (0, 80, 100)),
        MUADClass("motorcycle", 17, (0, 0, 230)),
        MUADClass("bicycle", 18, (119, 11, 32)),
        MUADClass("bear deer cow", 19, (255, 228, 196)),
        MUADClass("garbage_bag stand_food trash_can", 20, (128, 128, 0)),
        MUADClass("unlabeled", 21, (0, 0, 0)),  # id 255 or 21
    ]

    targets: list[Path] = []

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test", "ood"],
        version: Literal["small", "full"] = "full",
        min_depth: float | None = None,
        max_depth: float | None = None,
        target_type: Literal["semantic", "depth"] = "semantic",
        transforms: Callable | None = None,
        download: bool = False,
    ) -> None:
        """The MUAD Dataset.

        Args:
            root (str): Root directory of dataset where directory 'leftImg8bit'
                and 'leftLabel' or 'leftDepth' are located.
            split (str, optional): The image split to use, 'train' or 'val'.
            version (str, optional): The version of the dataset to use, 'small'
                or 'full'. Defaults to 'full'.
            min_depth (float, optional): The maximum depth value to use if
                target_type is 'depth'. Defaults to None.
            max_depth (float, optional): The maximum depth value to use if
                target_type is 'depth'. Defaults to None.
            target_type (str, optional): The type of target to use, 'semantic'
                or 'depth'.
            transforms (callable, optional): A function/transform that takes in
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
        if not cv2_installed:  # coverage: ignore
            raise ImportError(
                "The cv2 library is not installed. Please install"
                "torch_uncertainty with the image option:"
                """pip install -U "torch_uncertainty[image]"."""
            )

        if version == "small" and target_type == "depth":
            raise ValueError("Depth target is not available for the small version of MUAD.")

        logging.info(
            "MUAD is restricted to non-commercial use. By using MUAD, you "
            "agree to the terms and conditions."
        )

        dataset_root = Path(root) / "MUAD" if version == "full" else Path(root) / "MUAD_small"

        super().__init__(dataset_root, transforms=transforms)
        self.min_depth = min_depth
        self.max_depth = max_depth

        if split not in ["train", "val", "test", "ood"]:
            raise ValueError(f"split must be one of ['train', 'val']. Got {split}.")
        self.split = split
        self.version = version
        self.target_type = target_type

        if not self.check_split_integrity("leftImg8bit"):
            if download:
                self._download(split=split)
            else:
                raise FileNotFoundError(
                    f"MUAD {split} split not found or incomplete. Set download=True to download it."
                )

        if not self.check_split_integrity("leftLabel") and target_type == "semantic":
            if download:
                self._download(split=split)
            else:
                raise FileNotFoundError(
                    f"MUAD {split} split not found or incomplete. Set download=True to download it."
                )

        if not self.check_split_integrity("leftDepth") and target_type == "depth":
            if download:
                self._download(split=f"{split}_depth")
                # Depth target for train are in a different folder
                # thus we move them to the correct folder
                if split == "train":
                    shutil.move(
                        self.root / f"{split}_depth",
                        self.root / split / "leftDepth",
                    )
            else:
                raise FileNotFoundError(
                    f"MUAD {split} split not found or incomplete. Set download=True to download it."
                )

        self._make_dataset(self.root / split)

    def __getitem__(self, index: int) -> tuple[tv_tensors.Image, tv_tensors.Mask]:
        """Get the sample at the given index.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is either a segmentation mask
                or a depth map.
        """
        image = tv_tensors.Image(Image.open(self.samples[index]).convert("RGB"))
        if self.target_type == "semantic":
            target = tv_tensors.Mask(Image.open(self.targets[index]))
        else:
            os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
            target = Image.fromarray(
                cv2.imread(
                    str(self.targets[index]),
                    cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
                )
            )
            # TODO: in the long run it would be better to use a custom
            # tv_tensor for depth maps (e.g. tv_tensors.DepthMap)
            target = np.asarray(target, np.float32)
            target = tv_tensors.Mask(400 * (1 - target))  # convert to meters
            target[(target <= self.min_depth) | (target > self.max_depth)] = float("nan")

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def check_split_integrity(self, folder: str) -> bool:
        split_path = self.root / self.split
        return (
            split_path.is_dir() and len(list((split_path / folder).glob("**/*"))) == self.__len__()
        )

    def __len__(self) -> int:
        """The number of samples in the dataset."""
        return self._num_samples[self.version][self.split]

    def _make_dataset(self, path: Path) -> None:
        """Create a list of samples and targets.

        Args:
            path (Path): The path to the dataset.
        """
        if "depth" in path.name:
            raise NotImplementedError(
                "Depth mode is not implemented yet. Raise an issue if you need it."
            )
        self.samples = sorted((path / "leftImg8bit/").glob("**/*"))
        if self.target_type == "semantic":
            self.targets = sorted((path / "leftLabel/").glob("**/*"))
        elif self.target_type == "depth":
            self.targets = sorted((path / "leftDepth/").glob("**/*"))
        else:
            raise ValueError(
                f"target_type must be one of ['semantic', 'depth']. Got {self.target_type}."
            )

    def _download(self, split: str) -> None:
        """Download and extract the chosen split of the dataset."""
        if self.version == "small":
            filename = f"{split}.zip"
            downloaded_file = hf_hub_download(
                repo_id=self.small_muad_url, filename=filename, repo_type="dataset"
            )
            shutil.unpack_archive(downloaded_file, extract_dir=self.root)
        else:
            split_url = self.base_url + split + ".zip"
            download_and_extract_archive(split_url, self.root, md5=self.zip_md5[split])

    @property
    def color_palette(self) -> np.ndarray:
        return [c.color for c in self.classes]
