import re
from copy import deepcopy
from pathlib import Path

from torch import nn
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm

from torch_uncertainty.transforms.corruption import corruption_transforms


class CorruptedDataset(VisionDataset):
    def __init__(
        self,
        core_dataset: VisionDataset,
        shift_severity: int,
        generate: bool = False,
        on_the_fly: bool = False,
    ) -> None:
        """Generate the corrupted version of any VisionDataset.

        Args:
            core_dataset (VisionDataset): dataset to be corrupted.
            shift_severity (int): intensity of the corruption. Should be in [1, 5].
            generate (bool): Equivalent of the download attributes of the dataset. If ``True``,
                generate a new dataset with all the corrupted images. Defaults to ``False``.
            on_the_fly (bool): Generate the corrupted version of the dataset on the fly, without
                saving the images on disk. This is discouraged since the experiment won't be fully
                reproducible.

        Note:
            The corrupted dataset will use `transforms` of :attr:`core_dataset`.
        """
        super().__init__()
        self.core_dataset = core_dataset
        if shift_severity < 0:
            raise ValueError(f"Severity must be strictly greater than 0. Got {shift_severity}.")
        if not generate and on_the_fly:
            raise ValueError("generate must be True if on_the_fly is True.")

        self.shift_severity = shift_severity
        self.core_length = len(core_dataset)
        self.generate = generate
        self.on_the_fly = on_the_fly

        self.transforms = deepcopy(core_dataset.transforms)
        self.core_dataset.transform = None
        self.core_dataset.transforms = None
        self.core_dataset.target_transform = None

        dataset_name = str(type(core_dataset)).split(".")[-1][:-2]
        self.root = Path(core_dataset.root) / (dataset_name + "-C")

        if hasattr(self.core_dataset, "targets"):
            self.targets = self.core_dataset.targets
        elif hasattr(self.core_dataset, "labels"):
            self.targets = self.core_dataset.labels
        elif hasattr(self.core_dataset, "_labels"):
            self.targets = self.core_dataset._labels
        else:
            raise ValueError("The dataset should implement either targets, labels, or _labels.")

        self.targets = self.targets * len(corruption_transforms)

        if not generate:
            paths = sorted(self.root.glob(f"**/{self.shift_severity}/*.jpg"), key=lambda x: x.stem)
            self.samples = list(zip(paths, self.targets, strict=False))
            if len(paths) != 15 * self.core_length:
                raise ValueError(
                    "The corrupted dataset is not complete. Download it from HuggingFace or set generate=True."
                )

        if generate and not on_the_fly:
            self.root.mkdir(parents=True, exist_ok=True)
            self.to_tensor = ToTensor()
            self.to_pil = ToPILImage()
            self.samples = []

            self.prepare_data()

    def prepare_data(self):
        with logging_redirect_tqdm():
            pbar = tqdm(corruption_transforms)
            for corruption in pbar:
                corruption_name = re.sub(r"([a-z])([A-Z])", r"\1_\2", corruption.__name__).lower()
                pbar.set_description(f"Processing {corruption_name}")
                (self.root / corruption_name / f"{self.shift_severity}").mkdir(
                    parents=True, exist_ok=True
                )
                self._save_corruption(
                    self.root / corruption_name / f"{self.shift_severity}",
                    corruption(self.shift_severity),
                )

    def _save_corruption(self, root: Path, corruption: nn.Module) -> None:
        for i in trange(self.core_length, leave=False):
            img, tgt = self.core_dataset[i]
            img = corruption(self.to_tensor(img))
            self.to_pil(img).save(root / f"{i}.jpg")
            self.samples.append((root / f"{i}.jpg", tgt))
            self.targets.append(tgt)

    def __len__(self):
        """The length of the corrupted dataset."""
        return len(self.core_dataset) * len(corruption_transforms)

    def __getitem__(self, idx: int):
        """Get the corrupted image and the target.

        Args:
            idx (int): Index of the image to retrieve.
        """
        if self.on_the_fly:
            corrupt = corruption_transforms[idx // len(self.core_dataset)]
            idx = idx % len(self.core_dataset)
            img, target = self.core_dataset[idx]

            img = corrupt(img)
            if self.transforms is not None:
                img, target = self.transforms(img, target)

            return img, target

        path, target = self.samples[idx]
        img = default_loader(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target


if __name__ == "__main__":
    from torchvision.datasets import OxfordIIITPet

    dataset = OxfordIIITPet(root="data", split="test", download=True)
    corrupted_dataset = CorruptedDataset(dataset, shift_severity=5)
