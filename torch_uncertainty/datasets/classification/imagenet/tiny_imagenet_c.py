import logging
from collections.abc import Callable
from pathlib import Path

from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
)


class TinyImageNetC(ImageFolder):
    """The corrupted TinyImageNet-C Dataset.

    Args:
        root (str): Root directory of the datasets.
        transform (callable, optional): A function/transform that takes in
            a PIL image and returns a transformed version. E.g,
            ``transforms.RandomCrop``. Defaults to None.
        target_transform (callable, optional): A function/transform that
            takes in the target and transforms it. Defaults to None.
        subset (str): The subset to use, one of ``all`` or the keys in
            ``cifarc_subsets``.
        shift_severity (int): The shift_severity of the corruption, between 1 and 5.
        download (bool, optional): If True, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again. Defaults to False.

    References:
        Benchmarking neural network robustness to common corruptions and
            perturbations. Dan Hendrycks and Thomas Dietterich.
            In ICLR, 2019.
    """

    base_folder = "Tiny-ImageNet-C"
    tgz_md5 = [
        "f9c9a9dbdc11469f0b850190f7ad8be1",
        "0db0588d243cf403ef93449ec52b70eb",
    ]
    subsets = [
        "brightness",
        "contrast",
        "defocus_blur",
        "elastic_transform",
        "fog",
        "frost",
        "gaussian_blur",
        "gaussian_noise",
        "glass_blur",
        "impulse_noise",
        "jpeg_compression",
        "motion_blur",
        "pixelate",
        "saturate",
        "shot_noise",
        "snow",
        "spatter",
        "speckle_noise",
        "zoom_blur",
    ]

    url = "https://zenodo.org/record/8206060/files/"
    filename = ["Tiny-ImageNet-C.tar", "Tiny-ImageNet-C-extra.tar"]

    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        subset: str = "all",
        shift_severity: int = 1,
        download: bool = False,
    ) -> None:
        self.root = Path(root)

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found. You can use download=True to download it."
            )
        super().__init__(
            root=self.root / self.base_folder / "brightness/1/",
            transform=transform,
        )
        if subset not in ["all", *self.subsets]:
            raise ValueError(
                f"The subset '{subset}' does not exist in TinyImageNet-C."
            )
        self.subset = subset
        self.shift_severity = shift_severity

        self.transform = transform
        self.target_transform = target_transform

        if shift_severity not in list(range(1, 6)):
            raise ValueError(
                "Corruptions shift_severity should be chosen between 1 and 5 included."
            )

        # Update samples given the subset and shift_severity
        self._make_c_dataset(self.subset, self.shift_severity)

    def _make_c_dataset(self, subset: str, shift_severity: int) -> None:
        r"""Build the corrupted dataset according to the chosen subset and
            shift_severity. If the subset is 'all', gather all corruption types
            in the dataset.

        Args:
            subset (str): The name of the corruption subset to be used. Choose
                `all` for the dataset to contain all subsets.
            shift_severity (int): The shift_severity of the corruption applied to the
                images.
        """
        if subset == "all":
            collection = []
            for subset in self.subsets:
                imgs = [
                    (
                        img[0]
                        .replace("brightness", subset)
                        .replace("/1/", "/" + str(shift_severity) + "/"),
                        img[1],
                    )
                    for img in self.imgs
                ]

                collection.extend(imgs)
            self.imgs = collection
            self.samples = self.imgs
        else:
            self.imgs = [
                (
                    img[0]
                    .replace("brightness", subset)
                    .replace("/1/", "/" + str(shift_severity) + "/"),
                    img[1],
                )
                for img in self.imgs
            ]

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset."""
        for filename, md5 in list(
            zip(self.filename, self.tgz_md5, strict=True)
        ):
            if "extra" in filename:
                fpath = self.root / "Tiny-ImageNet-C" / filename
            else:
                fpath = self.root / filename
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        """Download the dataset."""
        if self._check_integrity():
            logging.info("Files already downloaded and verified")
            return
        for filename, md5 in list(
            zip(self.filename, self.tgz_md5, strict=True)
        ):
            if "extra" in filename:
                download_and_extract_archive(
                    self.url + filename,
                    self.root,
                    md5=md5,
                    filename="Tiny-ImageNet-C/" + filename,
                )
            else:
                download_and_extract_archive(
                    self.url + filename,
                    self.root,
                    md5=md5,
                )
