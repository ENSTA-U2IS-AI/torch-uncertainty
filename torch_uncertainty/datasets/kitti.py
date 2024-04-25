import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from PIL import Image
from torchvision import tv_tensors
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import (
    download_and_extract_archive,
)
from torchvision.transforms import functional as F
from tqdm import tqdm


class KITTIDepth(VisionDataset):
    depth_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_depth_annotated.zip"
    depth_md5 = "7d1ce32633dc2f43d9d1656a1f875e47"
    raw_url = "https://s3.eu-central-1.amazonaws.com/avg-kitti/raw_data/"
    raw_filenames = [
        "2011_09_26_calib.zip",
        "2011_09_26_drive_0001/2011_09_26_drive_0001_sync.zip",
        "2011_09_26_drive_0002/2011_09_26_drive_0002_sync.zip",
        "2011_09_26_drive_0005/2011_09_26_drive_0005_sync.zip",
        "2011_09_26_drive_0009/2011_09_26_drive_0009_sync.zip",
        "2011_09_26_drive_0011/2011_09_26_drive_0011_sync.zip",
        "2011_09_26_drive_0013/2011_09_26_drive_0013_sync.zip",
        "2011_09_26_drive_0014/2011_09_26_drive_0014_sync.zip",
        "2011_09_26_drive_0015/2011_09_26_drive_0015_sync.zip",
        "2011_09_26_drive_0017/2011_09_26_drive_0017_sync.zip",
        "2011_09_26_drive_0018/2011_09_26_drive_0018_sync.zip",
        "2011_09_26_drive_0019/2011_09_26_drive_0019_sync.zip",
        "2011_09_26_drive_0020/2011_09_26_drive_0020_sync.zip",
        "2011_09_26_drive_0022/2011_09_26_drive_0022_sync.zip",
        "2011_09_26_drive_0023/2011_09_26_drive_0023_sync.zip",
        "2011_09_26_drive_0027/2011_09_26_drive_0027_sync.zip",
        "2011_09_26_drive_0028/2011_09_26_drive_0028_sync.zip",
        "2011_09_26_drive_0029/2011_09_26_drive_0029_sync.zip",
        "2011_09_26_drive_0032/2011_09_26_drive_0032_sync.zip",
        "2011_09_26_drive_0035/2011_09_26_drive_0035_sync.zip",
        "2011_09_26_drive_0036/2011_09_26_drive_0036_sync.zip",
        "2011_09_26_drive_0039/2011_09_26_drive_0039_sync.zip",
        "2011_09_26_drive_0046/2011_09_26_drive_0046_sync.zip",
        "2011_09_26_drive_0048/2011_09_26_drive_0048_sync.zip",
        "2011_09_26_drive_0051/2011_09_26_drive_0051_sync.zip",
        "2011_09_26_drive_0052/2011_09_26_drive_0052_sync.zip",
        "2011_09_26_drive_0056/2011_09_26_drive_0056_sync.zip",
        "2011_09_26_drive_0057/2011_09_26_drive_0057_sync.zip",
        "2011_09_26_drive_0059/2011_09_26_drive_0059_sync.zip",
        "2011_09_26_drive_0060/2011_09_26_drive_0060_sync.zip",
        "2011_09_26_drive_0061/2011_09_26_drive_0061_sync.zip",
        "2011_09_26_drive_0064/2011_09_26_drive_0064_sync.zip",
        "2011_09_26_drive_0070/2011_09_26_drive_0070_sync.zip",
        "2011_09_26_drive_0079/2011_09_26_drive_0079_sync.zip",
        "2011_09_26_drive_0084/2011_09_26_drive_0084_sync.zip",
        "2011_09_26_drive_0086/2011_09_26_drive_0086_sync.zip",
        "2011_09_26_drive_0087/2011_09_26_drive_0087_sync.zip",
        "2011_09_26_drive_0091/2011_09_26_drive_0091_sync.zip",
        "2011_09_26_drive_0093/2011_09_26_drive_0093_sync.zip",
        "2011_09_26_drive_0095/2011_09_26_drive_0095_sync.zip",
        "2011_09_26_drive_0096/2011_09_26_drive_0096_sync.zip",
        "2011_09_26_drive_0101/2011_09_26_drive_0101_sync.zip",
        "2011_09_26_drive_0104/2011_09_26_drive_0104_sync.zip",
        "2011_09_26_drive_0106/2011_09_26_drive_0106_sync.zip",
        "2011_09_26_drive_0113/2011_09_26_drive_0113_sync.zip",
        "2011_09_26_drive_0117/2011_09_26_drive_0117_sync.zip",
        "2011_09_26_drive_0119/2011_09_26_drive_0119_sync.zip",
        "2011_09_28_calib.zip",
        "2011_09_28_drive_0001/2011_09_28_drive_0001_sync.zip",
        "2011_09_28_drive_0002/2011_09_28_drive_0002_sync.zip",
        "2011_09_28_drive_0016/2011_09_28_drive_0016_sync.zip",
        "2011_09_28_drive_0021/2011_09_28_drive_0021_sync.zip",
        "2011_09_28_drive_0034/2011_09_28_drive_0034_sync.zip",
        "2011_09_28_drive_0035/2011_09_28_drive_0035_sync.zip",
        "2011_09_28_drive_0037/2011_09_28_drive_0037_sync.zip",
        "2011_09_28_drive_0038/2011_09_28_drive_0038_sync.zip",
        "2011_09_28_drive_0039/2011_09_28_drive_0039_sync.zip",
        "2011_09_28_drive_0043/2011_09_28_drive_0043_sync.zip",
        "2011_09_28_drive_0045/2011_09_28_drive_0045_sync.zip",
        "2011_09_28_drive_0047/2011_09_28_drive_0047_sync.zip",
        "2011_09_28_drive_0053/2011_09_28_drive_0053_sync.zip",
        "2011_09_28_drive_0054/2011_09_28_drive_0054_sync.zip",
        "2011_09_28_drive_0057/2011_09_28_drive_0057_sync.zip",
        "2011_09_28_drive_0065/2011_09_28_drive_0065_sync.zip",
        "2011_09_28_drive_0066/2011_09_28_drive_0066_sync.zip",
        "2011_09_28_drive_0068/2011_09_28_drive_0068_sync.zip",
        "2011_09_28_drive_0070/2011_09_28_drive_0070_sync.zip",
        "2011_09_28_drive_0071/2011_09_28_drive_0071_sync.zip",
        "2011_09_28_drive_0075/2011_09_28_drive_0075_sync.zip",
        "2011_09_28_drive_0077/2011_09_28_drive_0077_sync.zip",
        "2011_09_28_drive_0078/2011_09_28_drive_0078_sync.zip",
        "2011_09_28_drive_0080/2011_09_28_drive_0080_sync.zip",
        "2011_09_28_drive_0082/2011_09_28_drive_0082_sync.zip",
        "2011_09_28_drive_0086/2011_09_28_drive_0086_sync.zip",
        "2011_09_28_drive_0087/2011_09_28_drive_0087_sync.zip",
        "2011_09_28_drive_0089/2011_09_28_drive_0089_sync.zip",
        "2011_09_28_drive_0090/2011_09_28_drive_0090_sync.zip",
        "2011_09_28_drive_0094/2011_09_28_drive_0094_sync.zip",
        "2011_09_28_drive_0095/2011_09_28_drive_0095_sync.zip",
        "2011_09_28_drive_0096/2011_09_28_drive_0096_sync.zip",
        "2011_09_28_drive_0098/2011_09_28_drive_0098_sync.zip",
        "2011_09_28_drive_0100/2011_09_28_drive_0100_sync.zip",
        "2011_09_28_drive_0102/2011_09_28_drive_0102_sync.zip",
        "2011_09_28_drive_0103/2011_09_28_drive_0103_sync.zip",
        "2011_09_28_drive_0104/2011_09_28_drive_0104_sync.zip",
        "2011_09_28_drive_0106/2011_09_28_drive_0106_sync.zip",
        "2011_09_28_drive_0108/2011_09_28_drive_0108_sync.zip",
        "2011_09_28_drive_0110/2011_09_28_drive_0110_sync.zip",
        "2011_09_28_drive_0113/2011_09_28_drive_0113_sync.zip",
        "2011_09_28_drive_0117/2011_09_28_drive_0117_sync.zip",
        "2011_09_28_drive_0119/2011_09_28_drive_0119_sync.zip",
        "2011_09_28_drive_0121/2011_09_28_drive_0121_sync.zip",
        "2011_09_28_drive_0122/2011_09_28_drive_0122_sync.zip",
        "2011_09_28_drive_0125/2011_09_28_drive_0125_sync.zip",
        "2011_09_28_drive_0126/2011_09_28_drive_0126_sync.zip",
        "2011_09_28_drive_0128/2011_09_28_drive_0128_sync.zip",
        "2011_09_28_drive_0132/2011_09_28_drive_0132_sync.zip",
        "2011_09_28_drive_0134/2011_09_28_drive_0134_sync.zip",
        "2011_09_28_drive_0135/2011_09_28_drive_0135_sync.zip",
        "2011_09_28_drive_0136/2011_09_28_drive_0136_sync.zip",
        "2011_09_28_drive_0138/2011_09_28_drive_0138_sync.zip",
        "2011_09_28_drive_0141/2011_09_28_drive_0141_sync.zip",
        "2011_09_28_drive_0143/2011_09_28_drive_0143_sync.zip",
        "2011_09_28_drive_0145/2011_09_28_drive_0145_sync.zip",
        "2011_09_28_drive_0146/2011_09_28_drive_0146_sync.zip",
        "2011_09_28_drive_0149/2011_09_28_drive_0149_sync.zip",
        "2011_09_28_drive_0153/2011_09_28_drive_0153_sync.zip",
        "2011_09_28_drive_0154/2011_09_28_drive_0154_sync.zip",
        "2011_09_28_drive_0155/2011_09_28_drive_0155_sync.zip",
        "2011_09_28_drive_0156/2011_09_28_drive_0156_sync.zip",
        "2011_09_28_drive_0160/2011_09_28_drive_0160_sync.zip",
        "2011_09_28_drive_0161/2011_09_28_drive_0161_sync.zip",
        "2011_09_28_drive_0162/2011_09_28_drive_0162_sync.zip",
        "2011_09_28_drive_0165/2011_09_28_drive_0165_sync.zip",
        "2011_09_28_drive_0166/2011_09_28_drive_0166_sync.zip",
        "2011_09_28_drive_0167/2011_09_28_drive_0167_sync.zip",
        "2011_09_28_drive_0168/2011_09_28_drive_0168_sync.zip",
        "2011_09_28_drive_0171/2011_09_28_drive_0171_sync.zip",
        "2011_09_28_drive_0174/2011_09_28_drive_0174_sync.zip",
        "2011_09_28_drive_0177/2011_09_28_drive_0177_sync.zip",
        "2011_09_28_drive_0179/2011_09_28_drive_0179_sync.zip",
        "2011_09_28_drive_0183/2011_09_28_drive_0183_sync.zip",
        "2011_09_28_drive_0184/2011_09_28_drive_0184_sync.zip",
        "2011_09_28_drive_0185/2011_09_28_drive_0185_sync.zip",
        "2011_09_28_drive_0186/2011_09_28_drive_0186_sync.zip",
        "2011_09_28_drive_0187/2011_09_28_drive_0187_sync.zip",
        "2011_09_28_drive_0191/2011_09_28_drive_0191_sync.zip",
        "2011_09_28_drive_0192/2011_09_28_drive_0192_sync.zip",
        "2011_09_28_drive_0195/2011_09_28_drive_0195_sync.zip",
        "2011_09_28_drive_0198/2011_09_28_drive_0198_sync.zip",
        "2011_09_28_drive_0199/2011_09_28_drive_0199_sync.zip",
        "2011_09_28_drive_0201/2011_09_28_drive_0201_sync.zip",
        "2011_09_28_drive_0204/2011_09_28_drive_0204_sync.zip",
        "2011_09_28_drive_0205/2011_09_28_drive_0205_sync.zip",
        "2011_09_28_drive_0208/2011_09_28_drive_0208_sync.zip",
        "2011_09_28_drive_0209/2011_09_28_drive_0209_sync.zip",
        "2011_09_28_drive_0214/2011_09_28_drive_0214_sync.zip",
        "2011_09_28_drive_0216/2011_09_28_drive_0216_sync.zip",
        "2011_09_28_drive_0220/2011_09_28_drive_0220_sync.zip",
        "2011_09_28_drive_0222/2011_09_28_drive_0222_sync.zip",
        "2011_09_28_drive_0225/2011_09_28_drive_0225_sync.zip",
        "2011_09_29_calib.zip",
        "2011_09_29_drive_0004/2011_09_29_drive_0004_sync.zip",
        "2011_09_29_drive_0026/2011_09_29_drive_0026_sync.zip",
        "2011_09_29_drive_0071/2011_09_29_drive_0071_sync.zip",
        "2011_09_29_drive_0108/2011_09_29_drive_0108_sync.zip",
        "2011_09_30_calib.zip",
        "2011_09_30_drive_0016/2011_09_30_drive_0016_sync.zip",
        "2011_09_30_drive_0018/2011_09_30_drive_0018_sync.zip",
        "2011_09_30_drive_0020/2011_09_30_drive_0020_sync.zip",
        "2011_09_30_drive_0027/2011_09_30_drive_0027_sync.zip",
        "2011_09_30_drive_0028/2011_09_30_drive_0028_sync.zip",
        "2011_09_30_drive_0033/2011_09_30_drive_0033_sync.zip",
        "2011_09_30_drive_0034/2011_09_30_drive_0034_sync.zip",
        "2011_09_30_drive_0072/2011_09_30_drive_0072_sync.zip",
        "2011_10_03_calib.zip",
        "2011_10_03_drive_0027/2011_10_03_drive_0027_sync.zip",
        "2011_10_03_drive_0034/2011_10_03_drive_0034_sync.zip",
        "2011_10_03_drive_0042/2011_10_03_drive_0042_sync.zip",
        "2011_10_03_drive_0047/2011_10_03_drive_0047_sync.zip",
        "2011_10_03_drive_0058/2011_10_03_drive_0058_sync.zip",
    ]

    _num_samples = {
        "train": 42949,
        "val": 3426,
        "test": ...,
    }

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val"],
        transforms: Callable | None = None,
        download: bool = False,
        remove_unused: bool = False,
    ) -> None:
        print(
            "KITTIDepth is copyrighted by the Karlsruhe Institute of Technology "
            "(KIT) and the Toyota Technological Institute at Chicago (TTIC). "
            "By using KITTIDepth, you agree to the terms and conditions of the "
            "Creative Commons Attribution-NonCommercial-ShareAlike 3.0 License. "
            "This means that you must attribute the work in the manner specified "
            "by the authors, you may not use this work for commercial purposes "
            "and if you alter, transform, or build upon this work, you may "
            "distribute the resulting work only under the same license."
        )

        super().__init__(
            root=Path(root) / "KITTIDepth",
            transforms=transforms,
        )

        if split not in ["train", "val"]:
            raise ValueError(
                f"split must be one of ['train', 'val']. Got {split}."
            )

        self.split = split

        if not self.check_split_integrity("leftDepth"):
            if download:
                self._download_depth()
            else:
                raise FileNotFoundError(
                    f"KITTI {split} split not found or incomplete. Set download=True to download it."
                )

        if not self.check_split_integrity("leftImg8bit"):
            if download:
                self._download_raw(remove_unused)
            else:
                raise FileNotFoundError(
                    f"KITTI {split} split not found or incomplete. Set download=True to download it."
                )

        self._make_dataset()

    def check_split_integrity(self, folder: str) -> bool:
        split_path = self.root / self.split
        return (
            split_path.is_dir()
            and len(list((split_path / folder).glob("*.png")))
            == self._num_samples[self.split]
        )

    def __getitem__(
        self, index: int
    ) -> tuple[tv_tensors.Image, tv_tensors.Mask]:
        """Get the sample at the given index.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is a depth map.
        """
        image = tv_tensors.Image(Image.open(self.samples[index]).convert("RGB"))
        target = tv_tensors.Mask(
            F.pil_to_tensor(Image.open(self.targets[index])).squeeze(0) / 256.0
        )
        target[target == 0.0] = float("nan")

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        """The number of samples in the dataset."""
        return self._num_samples[self.split]

    def _make_dataset(self) -> None:
        self.samples = sorted(
            (self.root / self.split / "leftImg8bit").glob("*.png")
        )
        self.targets = sorted(
            (self.root / self.split / "leftDepth").glob("*.png")
        )

    def _download_depth(self) -> None:
        """Download and extract the depth annotation dataset."""
        if not (self.root / "tmp").exists():
            download_and_extract_archive(
                self.depth_url,
                download_root=self.root,
                extract_root=self.root / "tmp",
                md5=self.depth_md5,
            )

        print("Structuring the dataset depth annotations...")

        if (self.root / "train" / "leftDepth").exists():
            shutil.rmtree(self.root / "train" / "leftDepth")

        (self.root / "train" / "leftDepth").mkdir(parents=True, exist_ok=False)

        depth_files = list((self.root).glob("**/tmp/train/**/image_02/*.png"))
        print("Train files:")
        for file in tqdm(depth_files):
            exp_code = file.parents[3].name.split("_")
            filecode = "_".join(
                [exp_code[0], exp_code[1], exp_code[2], exp_code[4], file.name]
            )
            shutil.copy(file, self.root / "train" / "leftDepth" / filecode)

        if (self.root / "val" / "leftDepth").exists():
            shutil.rmtree(self.root / "val" / "leftDepth")

        (self.root / "val" / "leftDepth").mkdir(parents=True, exist_ok=False)

        depth_files = list((self.root).glob("**/tmp/val/**/image_02/*.png"))
        print("Validation files:")
        for file in tqdm(depth_files):
            exp_code = file.parents[3].name.split("_")
            filecode = "_".join(
                [exp_code[0], exp_code[1], exp_code[2], exp_code[4], file.name]
            )
            shutil.copy(file, self.root / "val" / "leftDepth" / filecode)

        shutil.rmtree(self.root / "tmp")

    def _download_raw(self, remove_unused: bool) -> None:
        """Download and extract the raw dataset."""
        for filename in self.raw_filenames:
            print(self.raw_url + filename)
            download_and_extract_archive(
                self.raw_url + filename,
                download_root=self.root,
                extract_root=self.root / "raw",
                md5=None,
            )

        print("Structuring the dataset raw data...")

        samples_to_keep = list(
            (self.root / "train" / "leftDepth").glob("*.png")
        )

        if (self.root / "train" / "leftImg8bit").exists():
            shutil.rmtree(self.root / "train" / "leftImg8bit")

        (self.root / "train" / "leftImg8bit").mkdir(
            parents=True, exist_ok=False
        )

        print("Train files:")
        for sample in tqdm(samples_to_keep):
            filecode = sample.name.split("_")
            first_level = "_".join([filecode[0], filecode[1], filecode[2]])
            second_level = "_".join(
                [
                    filecode[0],
                    filecode[1],
                    filecode[2],
                    "drive",
                    filecode[3],
                    "sync",
                ]
            )
            raw_path = (
                self.root
                / "raw"
                / first_level
                / second_level
                / "image_02"
                / "data"
                / filecode[4]
            )
            shutil.copy(
                raw_path, self.root / "train" / "leftImg8bit" / sample.name
            )

        samples_to_keep = list((self.root / "val" / "leftDepth").glob("*.png"))

        if (self.root / "val" / "leftImg8bit").exists():
            shutil.rmtree(self.root / "val" / "leftImg8bit")

        (self.root / "val" / "leftImg8bit").mkdir(parents=True, exist_ok=False)

        print("Validation files:")
        for sample in tqdm(samples_to_keep):
            filecode = sample.name.split("_")
            first_level = "_".join([filecode[0], filecode[1], filecode[2]])
            second_level = "_".join(
                [
                    filecode[0],
                    filecode[1],
                    filecode[2],
                    "drive",
                    filecode[3],
                    "sync",
                ]
            )
            raw_path = (
                self.root
                / "raw"
                / first_level
                / second_level
                / "image_02"
                / "data"
                / filecode[4]
            )
            shutil.copy(
                raw_path, self.root / "val" / "leftImg8bit" / sample.name
            )

        if remove_unused:
            shutil.rmtree(self.root / "raw")
