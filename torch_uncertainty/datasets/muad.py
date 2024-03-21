import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from einops import rearrange
from PIL import Image
from torchvision import tv_tensors
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    download_url,
)
from torchvision.transforms.v2 import functional as F


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
        target_type: Literal["semantic", "depth"] = "semantic",
        transforms: Callable | None = None,
        download: bool = False,
    ) -> None:
        """The MUAD Dataset.

        Args:
            root (str): Root directory of dataset where directory 'leftImg8bit'
                and 'leftLabel' are located.
            split (str, optional): The image split to use, 'train', 'val',
            'train_depth' or 'val_depth'.
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
        print(
            "MUAD is restricted to non-commercial use. By using MUAD, you "
            "agree to the terms and conditions."
        )
        super().__init__(
            root=Path(root) / "MUAD",
            transforms=transforms,
        )

        if split not in ["train", "val"]:
            raise ValueError(
                f"split must be one of ['train', 'val']. Got {split}."
            )
        self.split = split
        self.target_type = target_type

        split_path = self.root / (split + ".zip")
        if (not check_integrity(split_path, self.zip_md5[split])) and download:
            self._download(split=self.split)

        if (
            self.target_type == "depth"
            and not check_integrity(split_path, self.zip_md5[split + "_depth"])
            and download
        ):
            self._download(split=f"{split}_depth")

        # Load classes metadata
        cls_path = self.root / "classes.json"
        if (not check_integrity(cls_path, self.classes_md5)) and download:
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

    def encode_target(self, target: Image.Image) -> Image.Image:
        """Encode target image to tensor.

        Args:
            target (Image.Image): Target PIL image.

        Returns:
            torch.Tensor: Encoded target.
        """
        target = F.pil_to_tensor(target)
        target = rearrange(target, "c h w -> h w c")
        out = torch.zeros_like(target[..., :1])
        # convert target color to index
        for muad_class in self.classes:
            out[
                (
                    target == torch.tensor(muad_class["id"], dtype=target.dtype)
                ).all(dim=-1)
            ] = muad_class["train_id"]

        return F.to_pil_image(rearrange(out, "h w c -> c h w"))

    def decode_target(self, target: Image.Image) -> np.ndarray:
        target[target == 255] = 19
        return self.train_id_to_color[target]

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """Get the sample at the given index.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is either a segmentation mask
                or a depth map.
        """
        image = tv_tensors.Image(Image.open(self.samples[index]).convert("RGB"))
        target = tv_tensors.Mask(
            self.encode_target(Image.open(self.targets[index]))
        )

        if self.transforms is not None:
            image, target = self.transforms(image, target)

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
        if self.target_type == "semantic":
            self.targets = list((path / "leftLabel/").glob("**/*"))
        else:
            raise NotImplementedError(
                "Depth regression mode is not implemented yet. Raise an issue "
                "if you need it."
            )

    def _download(self, split: str):
        """Download and extract the chosen split of the dataset."""
        split_url = self.base_url + split + ".zip"
        download_and_extract_archive(
            split_url, self.root, md5=self.zip_md5[split]
        )
