import shutil
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from PIL import Image
from torchvision import tv_tensors
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_and_extract_archive


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
    }
    filenames = {
        "raw": "701_StillsRaw_full.zip",
        "label": "LabeledApproved_full.zip",
    }
    base_folder = "camvid"
    num_samples = 701

    def __init__(
        self,
        root: str,
        transforms: Callable | None = None,
        download: bool = False,
    ) -> None:
        """`CamVid <http://web4.cs.ucl.ac.uk/staff/g.brostow/MotionSegRecData/>`_ Dataset.

        Args:
            root (str): Root directory of dataset where ``camvid/`` exists or
                will be saved to if download is set to ``True``.
            transforms (callable, optional): A function/transform that takes
                input sample and its target as entry and returns a transformed
                version.
            download (bool, optional): If true, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again.
        """
        super().__init__(root, transforms, None, None)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. "
                "You can use download=True to download it"
            )

        self.images = sorted((Path(self.root) / "camvid" / "raw").glob("*.png"))
        self.targets = sorted(
            (Path(self.root) / "camvid" / "label").glob("*.png")
        )

    def __getitem__(self, index: int) -> tuple:
        """Get image and target at index.

        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is the segmentation mask.
        """
        image = tv_tensors.Image(Image.open(self.images[index]).convert("RGB"))
        target = tv_tensors.Mask(Image.open(self.targets[index]))

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        """Return the number of samples."""
        return self.num_samples

    def _check_integrity(self) -> bool:
        """Check if the dataset exists."""
        if (
            len(list((Path(self.root) / "camvid" / "raw").glob("*.png")))
            != self.num_samples
        ):
            return False
        if (
            len(list((Path(self.root) / "camvid" / "label").glob("*.png")))
            != self.num_samples
        ):
            return False
        return True

    def download(self) -> None:
        """Download the CamVid data if it doesn't exist already."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

        if Path(self.root) / self.base_folder:
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


if __name__ == "__main__":
    dataset = CamVid("data", download=True)
    print(dataset)
