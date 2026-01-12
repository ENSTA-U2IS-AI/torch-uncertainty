import logging
import os
import shutil
from collections.abc import Callable
from importlib import util
from operator import attrgetter
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


class MUADClass(NamedTuple):
    name: str
    id: int
    train_id: int
    color: tuple[int, int, int]
    is_ood: bool


class MUAD(VisionDataset):
    classes_url = "https://raw.githubusercontent.com/torch-uncertainty/dataset-metadata/main/segmentation/muad/classes.json"

    base_urls = {
        "full": "ENSTA-U2IS/MUAD",
        "small": "ENSTA-U2IS/miniMUAD",
    }
    huggingface_splits = {
        "full": [
            "train",
            "val",
            "test_id",
            "test_ood",
            "test_id_low_adv",
            "test_id_high_adv",
            "test_ood_low_adv",
            "test_ood_high_adv",
        ],
        "small": [
            "train",
            "val",
            "test",
            "ood",
        ],
    }

    _num_samples = {
        "full": {
            "train": 3420,
            "val": 492,
            "test_id": 551,
            "test_ood": 1668,
            "test_id_low_adv": 605,
            "test_id_high_adv": 602,
            "test_ood_low_adv": 1552,
            "test_ood_high_adv": 1421,
        },
        "small": {
            "train": 400,
            "val": 54,
            "test": 112,
            "ood": 20,
        },
    }

    classes = [
        MUADClass("road", 0, 0, (128, 64, 128), False),
        MUADClass("sidewalk", 1, 1, (244, 35, 232), False),
        MUADClass("building", 2, 2, (70, 70, 70), False),
        MUADClass("wall", 3, 3, (102, 102, 156), False),
        MUADClass("fence", 4, 4, (190, 153, 153), False),
        MUADClass("pole", 5, 5, (153, 153, 153), False),
        MUADClass("traffic_light", 6, 6, (250, 170, 30), False),
        MUADClass("traffic_sign", 7, 7, (220, 220, 0), False),
        MUADClass("vegetation", 8, 8, (107, 142, 35), False),
        MUADClass("terrain", 9, 9, (152, 251, 152), False),
        MUADClass("sky", 10, 10, (70, 130, 180), False),
        MUADClass("person", 11, 11, (220, 20, 60), False),
        MUADClass("car", 13, 12, (0, 0, 142), False),
        MUADClass("truck", 14, 13, (0, 0, 70), False),
        MUADClass("bus", 15, 14, (0, 60, 100), False),
        MUADClass("rider", 12, 15, (255, 0, 0), True),
        MUADClass("train", 16, 16, (0, 80, 100), True),
        MUADClass("motorcycle", 17, 17, (0, 0, 230), True),
        MUADClass("bicycle", 18, 18, (119, 11, 32), True),
        MUADClass("bear deer cow", 19, 19, (255, 228, 196), True),
        MUADClass("garbage_bag stand_food trash_can", 20, 20, (128, 128, 0), True),
        MUADClass("unlabeled", 21, 255, (0, 0, 0), False),  # id 255 or 21
    ]

    targets: list[Path] = []

    num_id_classes = 15
    num_ood_classes = 6

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
        use_train_ids: bool = True,
    ) -> None:
        """The MUAD Dataset.

        Args:
            root (str | Path): Root directory of dataset where directory ``leftImg8bit`` and ``leftLabel``
                or ``leftDepth`` are located.
            split (str, optional): The image split to use, ``train``, ``val``, ``test`` or ``ood``.
            version (str, optional): The version of the dataset to use, ``small`` or ``full``.
                Defaults to ``full``.
            min_depth (float, optional): The maximum depth value to use if target_type is ``depth``.
                Defaults to ``None``.
            max_depth (float, optional): The maximum depth value to use if target_type is ``depth``.
                Defaults to ``None``.
            target_type (str, optional): The type of target to use, ``semantic`` or ``depth``.
                Defaults to ``semantic``.
            transforms (callable, optional): A function/transform that takes in a tuple of PIL
                images and returns a transformed version. Defaults to ``None``.
            download (bool, optional): If ``True``, downloads the dataset from the internet and puts
                it in root directory. If dataset is already downloaded, it is not downloaded again.
                Defaults to ``False``.
            use_train_ids (bool, optional): If ``True``, uses the train ids instead of the original
                ids. Defaults to ``True``. Note that this is only used for the ``semantic`` target
                type.

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

        if split not in self.huggingface_splits[version]:
            raise ValueError(
                f"split must be one of {self.huggingface_splits[version]}. Got {split}."
            )
        self.split = split
        self.version = version
        self.target_type = target_type
        self.use_train_ids = use_train_ids

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

    def encode_target(self, target: tv_tensors.Mask) -> tv_tensors.Mask:
        """Encode the target tensor to the train ids.

        Args:
            target (torch.Tensor): Target tensor.

        Returns:
            torch.Tensor: Encoded target tensor.
        """
        original_tgt = target.clone()
        for c in self.classes:
            target[original_tgt == c.id] = c.train_id
        return target

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
            if self.use_train_ids:
                target = self.encode_target(target)
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

    def _download(self, split: str) -> None:  # coverage: ignore
        """Download and extract the chosen split of the dataset."""
        repo_id = self.base_urls[self.version]
        filename = f"{split}.zip"

        downloaded_file = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        shutil.unpack_archive(downloaded_file, extract_dir=self.root)

    @property
    def color_palette(self) -> np.ndarray:
        sorting_key = "train_id" if self.use_train_ids else "id"
        sorted_cls = sorted(self.classes, key=attrgetter(sorting_key))
        return [c.color for c in sorted_cls]
