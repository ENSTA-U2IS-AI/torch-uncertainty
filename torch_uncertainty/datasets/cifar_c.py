# fmt:off
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from torchvision.datasets import VisionDataset

import numpy as np

# fmt:on


class CIFAR10_C(VisionDataset):
    cifarc_subsets = [
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

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        dataset_path: Path = Path("CIFAR-10-C"),
        subset: str = "brightness",
        severity: int = 1,
    ):
        super().__init__(root=root / dataset_path, transform=transform)
        assert (
            subset in ["all"] + self.cifarc_subsets
        ), f"The subset '{subset}' does not exist in CIFAR-C."
        self.subset = subset
        self.severity = severity

        self.transform = transform
        self.target_transform = target_transform

        assert severity in list(
            range(1, 6)
        ), "Corruptions severity should be chosen between 1 and 5 included."
        samples, labels = self.make_dataset(
            self.root, self.subset, self.severity
        )

        self.samples = samples
        self.labels = labels

    def make_dataset(
        self, root: Path, subset: str, severity: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        r"""
        Build the corrupted dataset according to the chosen subset and
            severity. If the subset is 'all', gather all corruption types
            in the dataset.
        Args:
            root (Path):The path to the dataset.
            subset (str): The name of the corruption subset to be used. Choose
                `all` for the dataset to contain all subsets.
            severity (int): The severity of the corruption applied to the
                images.
        Returns:
            Tuple[np.ndarray, np.ndarray]: The samples and labels of the chosen
        """
        if subset == "all":
            sample_arrays = []
            labels: np.ndarray = np.load(root / "labels.npy")[
                (severity - 1) * 10000 : severity * 10000
            ]
            for cifar_subset in self.cifarc_subsets:
                sample_arrays.append(
                    np.load(root / (cifar_subset + ".npy"))[
                        (severity - 1) * 10000 : severity * 10000
                    ]
                )
            samples = np.concatenate(sample_arrays, axis=0)
            labels = np.tile(labels, len(self.cifarc_subsets))

        else:
            samples: np.ndarray = np.load(root / (subset + ".npy"))[
                (severity - 1) * 10000 : severity * 10000
            ]
            labels: np.ndarray = np.load(root / "labels.npy")[
                (severity - 1) * 10000 : severity * 10000
            ]
        return samples, labels

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> Any:
        sample, target = (
            self.samples[index],
            self.labels[index],
        )

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


class CIFAR100_C(CIFAR10_C):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        dataset_path: Path = Path("CIFAR-100-C"),
        subset: str = "brightness",
        severity: int = 1,
    ):
        super().__init__(
            root,
            transform=transform,
            dataset_path=dataset_path,
            subset=subset,
            severity=severity,
        )
