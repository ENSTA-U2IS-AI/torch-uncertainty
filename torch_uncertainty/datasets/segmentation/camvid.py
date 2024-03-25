import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Literal, NamedTuple

import torch
from einops import rearrange, repeat
from PIL import Image
from torchvision import tv_tensors
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    download_and_extract_archive,
    download_url,
)
from torchvision.transforms.v2 import functional as F


class CamVidClass(NamedTuple):
    name: str
    index: int
    color: tuple[int, int, int]


class CamVid(VisionDataset):
    # Notes: some classes are not used here
    classes = [
        CamVidClass("sky", 0, (128, 128, 128)),
        CamVidClass("building", 1, (128, 0, 0)),
        CamVidClass("pole", 2, (192, 192, 128)),
        CamVidClass("road_marking", 3, (255, 69, 0)),
        CamVidClass("road", 4, (128, 64, 128)),
        CamVidClass("pavement", 5, (60, 40, 222)),
        CamVidClass("tree", 6, (128, 128, 0)),
        CamVidClass("sign_symbol", 7, (192, 128, 128)),
        CamVidClass("fence", 8, (64, 64, 128)),
        CamVidClass("car", 9, (64, 0, 128)),
        CamVidClass("pedestrian", 10, (64, 64, 0)),
        CamVidClass("bicyclist", 11, (0, 128, 192)),
        CamVidClass("unlabelled", 12, (0, 0, 0)),
    ]

    urls = {
        "raw": "http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/files/701_StillsRaw_full.zip",
        "label": "http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/data/LabeledApproved_full.zip",
        "splits": "https://raw.githubusercontent.com/torch-uncertainty/dataset-metadata/main/segmentation/camvid/splits.json",
    }

    splits_md5 = "db45289aaa83c60201391b11e78c6382"

    filenames = {
        "raw": "701_StillsRaw_full.zip",
        "label": "LabeledApproved_full.zip",
    }
    base_folder = "camvid"
    num_samples = {
        "train": 367,
        "val": 101,
        "test": 233,
        "all": 701,
    }

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] | None = None,
        transforms: Callable | None = None,
        download: bool = False,
    ) -> None:
        """`CamVid <http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/>`_ Dataset.

        Args:
            root (str): Root directory of dataset where ``camvid/`` exists or
                will be saved to if download is set to ``True``.
            split (str, optional): The dataset split, supports ``train``,
                ``val`` and ``test``. Default: ``None``.
            transforms (callable, optional): A function/transform that takes
                input sample and its target as entry and returns a transformed
                version. Default: ``None``.
            download (bool, optional): If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
        """
        if split not in ["train", "val", "test", None]:
            raise ValueError(
                f"Unknown split '{split}'. "
                "Supported splits are ['train', 'val', 'test', None]"
            )

        super().__init__(root, transforms, None, None)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                "You can use download=True to download it"
            )

        # get filenames for split
        if split is None:
            self.images = sorted(
                (Path(self.root) / "camvid" / "raw").glob("*.png")
            )
            self.targets = sorted(
                (Path(self.root) / "camvid" / "label").glob("*.png")
            )
        else:
            with (Path(self.root) / "camvid" / "splits.json").open() as f:
                filenames = json.load(f)[split]

            self.images = sorted(
                [
                    path
                    for path in (Path(self.root) / "camvid" / "raw").glob(
                        "*.png"
                    )
                    if path.stem in filenames
                ]
            )
            self.targets = sorted(
                [
                    path
                    for path in (Path(self.root) / "camvid" / "label").glob(
                        "*.png"
                    )
                    if path.stem[:-2] in filenames
                ]
            )

        self.split = split if split is not None else "all"

    def encode_target(self, target: Image.Image) -> torch.Tensor:
        """Encode target image to tensor.

        Args:
            target (Image.Image): Target PIL image.

        Returns:
            torch.Tensor: Encoded target.
        """
        colored_target = F.pil_to_tensor(target)
        colored_target = rearrange(colored_target, "c h w -> h w c")
        target = torch.zeros_like(colored_target[..., :1])
        # convert target color to index
        for camvid_class in self.classes:
            index = camvid_class.index if camvid_class.index != 12 else 255
            target[
                (
                    colored_target
                    == torch.tensor(camvid_class.color, dtype=target.dtype)
                ).all(dim=-1)
            ] = index

        return rearrange(target, "h w c -> c h w")

    def decode_target(self, target: torch.Tensor) -> Image.Image:
        """Decode target tensor to image.

        Args:
            target (torch.Tensor): Target tensor.

        Returns:
            Image.Image: Decoded target as a PIL.Image.
        """
        colored_target = repeat(target.clone(), "h w  -> h w 3", c=3)

        for camvid_class in self.classes:
            colored_target[
                (
                    target
                    == torch.tensor(camvid_class.index, dtype=target.dtype)
                ).all(dim=0)
            ] = torch.tensor(camvid_class.color, dtype=target.dtype)

        return F.to_pil_image(rearrange(colored_target, "h w c -> c h w"))

    def __getitem__(
        self, index: int
    ) -> tuple[tv_tensors.Image, tv_tensors.Mask]:
        """Get the image and target at the given index.

        Args:
            index (int): Sample index.

        Returns:
            tuple[tv_tensors.Image, tv_tensors.Mask]: Image and target.
        """
        image = tv_tensors.Image(Image.open(self.images[index]).convert("RGB"))
        target = tv_tensors.Mask(
            self.encode_target(Image.open(self.targets[index]))
        )

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_samples[self.split]

    def _check_integrity(self) -> bool:
        """Check if the dataset exists."""
        if (
            len(list((Path(self.root) / "camvid" / "raw").glob("*.png")))
            != self.num_samples["all"]
        ):
            return False
        if (
            len(list((Path(self.root) / "camvid" / "label").glob("*.png")))
            != self.num_samples["all"]
        ):
            return False

        return (Path(self.root) / "camvid" / "splits.json").exists()

    def download(self) -> None:
        """Download the CamVid data if it doesn't exist already."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        if (Path(self.root) / self.base_folder).exists():
            shutil.rmtree(Path(self.root) / self.base_folder)

        download_and_extract_archive(
            self.urls["raw"],
            self.root,
            extract_root=Path(self.root) / "camvid",
            filename=self.filenames["raw"],
        )
        (Path(self.root) / "camvid" / "701_StillsRaw_full").replace(
            Path(self.root) / "camvid" / "raw"
        )
        download_and_extract_archive(
            self.urls["label"],
            self.root,
            extract_root=Path(self.root) / "camvid" / "label",
            filename=self.filenames["label"],
        )
        download_url(
            self.urls["splits"],
            Path(self.root) / "camvid",
            filename="splits.json",
            md5=self.splits_md5,
        )
