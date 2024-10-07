from collections.abc import Callable
from importlib import util
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image
from torchvision import tv_tensors
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    download_url,
)

if util.find_spec("cv2"):
    import cv2

    cv2_installed = True
else:  # coverage: ignore
    cv2_installed = False

if util.find_spec("h5py"):
    import h5py

    h5py_installed = True
else:  # coverage: ignore
    h5py_installed = False


class NYUv2(VisionDataset):
    root: Path
    rgb_urls = {
        "train": "http://www.doc.ic.ac.uk/~ahanda/nyu_train_rgb.tgz",
        "val": "http://www.doc.ic.ac.uk/~ahanda/nyu_test_rgb.tgz",
    }
    rgb_md5 = {
        "train": "ad124bbde47e371359caa4642a8a4611",
        "val": "f47f7c7c8a20d1210db7941c4f153b06",
    }
    depth_url = "http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat"
    depth_md5 = "520609c519fba3ba5ac58c8fefcc3530"

    def __init__(
        self,
        root: Path | str,
        split: Literal["train", "val"],
        transforms: Callable | None = None,
        min_depth: float = 0.0,
        max_depth: float = 10.0,
        download: bool = False,
    ):
        """NYUv2 depth dataset.

        Args:
            root (Path | str): Root directory where dataset is stored.
            split (Literal["train", "val"]): Dataset split.
            transforms (Callable | None): Transform to apply to samples & targets.
                Defaults to None.
            min_depth (float): Minimum depth value. Defaults to 1e-3.
            max_depth (float): Maximum depth value. Defaults to 10.
            download (bool): Download dataset if not found. Defaults to False.
        """
        if not cv2_installed:  # coverage: ignore
            raise ImportError(
                "The cv2 library is not installed. Please install"
                "torch_uncertainty with the image option:"
                """pip install -U "torch_uncertainty[image]"."""
            )
        if not h5py_installed:  # coverage: ignore
            raise ImportError(
                "The h5py library is not installed. Please install"
                "torch_uncertainty with the image option:"
                """pip install -U "torch_uncertainty[image]"."""
            )

        super().__init__(Path(root) / "NYUv2", transforms=transforms)
        self.min_depth = min_depth
        self.max_depth = max_depth

        if split not in ["train", "val"]:
            raise ValueError(
                f"split must be one of ['train', 'val']. Got {split}."
            )
        self.split = split

        if not self._check_integrity():
            if download:
                self._download()
            else:
                raise FileNotFoundError(
                    f"NYUv2 {split} split not found or incomplete. Set download=True to download it."
                )

        # make dataset
        path = self.root / self.split
        self.samples = sorted((path / "rgb_img").glob("**/*"))
        self.targets = sorted((path / "depth").glob("**/*"))

    def __getitem__(self, index: int):
        """Return image and target at index.

        Args:
            index (int): Index of the sample.
        """
        image = tv_tensors.Image(Image.open(self.samples[index]).convert("RGB"))
        target = Image.fromarray(
            cv2.imread(
                str(self.targets[index]),
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
            )
        )
        target = np.asarray(target, np.uint16)
        target = tv_tensors.Mask(target / 1e4)  # convert to meters
        target[(target <= self.min_depth) | (target > self.max_depth)] = float(
            "nan"
        )
        if self.transforms is not None:
            image, target = self.transforms(image, target)
        return image, target

    def __len__(self):
        """Return number of samples in dataset."""
        return len(self.samples)

    def _check_integrity(self) -> bool:
        """Check if dataset is present and complete."""
        return (
            check_integrity(
                self.root / f"nyu_{self.split}_rgb.tgz",
                self.rgb_md5[self.split],
            )
            and check_integrity(self.root / "depth.mat", self.depth_md5)
            and (self.root / self.split / "rgb_img").exists()
            and (self.root / self.split / "depth").exists()
        )

    def _download(self):
        """Download and extract dataset."""
        download_and_extract_archive(
            self.rgb_urls[self.split],
            self.root,
            extract_root=self.root / self.split / "rgb_img",
            filename=f"nyu_{self.split}_rgb.tgz",
            md5=self.rgb_md5[self.split],
        )
        if not check_integrity(self.root / "depth.mat", self.depth_md5):
            download_url(
                NYUv2.depth_url, self.root, "depth.mat", self.depth_md5
            )
        self._create_depth_files()

    def _create_depth_files(self):
        """Create depth images from the depth.mat file."""
        path = self.root / self.split
        (path / "depth").mkdir()
        samples = sorted((path / "rgb_img").glob("**/*"))
        ids = [int(p.stem.split("_")[-1]) for p in samples]
        file = h5py.File(self.root / "depth.mat", "r")
        depths = file["depths"]
        for i in range(len(depths)):
            img_id = i + 1
            if img_id in ids:
                img = (depths[i] * 1e4).astype(np.uint16).T
                Image.fromarray(img).save(
                    path / "depth" / f"nyu_depth_{str(img_id).zfill(4)}.png"
                )
