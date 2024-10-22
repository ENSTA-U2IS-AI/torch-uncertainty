from copy import deepcopy
from pathlib import Path

from PIL import Image
from torch import nn
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToPILImage, ToTensor
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from torch_uncertainty.transforms.corruption import corruption_transforms


class CorruptedDataset(VisionDataset):
    def __init__(
        self,
        core_dataset: VisionDataset,
        shift_severity: int,
        on_the_fly: bool = False,
    ) -> None:
        super().__init__()
        self.core_dataset = core_dataset
        if shift_severity <= 0:
            raise ValueError(
                f"Severity must be greater than 0. Got {shift_severity}."
            )
        self.shift_severity = shift_severity
        self.core_length = len(core_dataset)
        self.on_the_fly = on_the_fly
        self.transforms = deepcopy(core_dataset.transforms)
        self.target_transforms = deepcopy(core_dataset.target_transform)
        self.core_dataset.transform = None
        self.core_dataset.target_transform = None

        self.root = Path(core_dataset.root)
        dataset_name = str(type(core_dataset)).split(".")[-1][:-2].lower()
        self.root /= dataset_name + "_corrupted"
        self.root /= f"severity_{self.shift_severity}"
        self.root.mkdir(parents=True)

        if not on_the_fly:
            self.to_tensor = ToTensor()
            self.to_pil = ToPILImage()
            self.samples = []
            self.targets = self.core_dataset.targets * 10
            self.prepare_data()

    def prepare_data(self):
        with logging_redirect_tqdm():
            for corruption in tqdm(corruption_transforms):
                corruption_name = corruption.__name__.lower()
                (self.root / corruption_name).mkdir(parents=True)
                self.save_corruption(
                    self.root / corruption_name, corruption(self.shift_severity)
                )

    def save_corruption(self, root: Path, corruption: nn.Module) -> None:
        for i in range(self.core_length):
            img, tgt = self.core_dataset[i]
            if isinstance(img, str | Path):
                img = Image.open(img).convert("RGB")
            img = corruption(self.to_tensor(img))
            self.to_pil(img).save(root / f"{i}.png")
            self.samples.append(root / f"{i}.png")
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
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)
            return img, target

        img, target = self.core_dataset[idx]
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


if __name__ == "__main__":
    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(root="data", download=True)
    corrupted_dataset = CorruptedDataset(dataset, shift_severity=1)
    print(len(corrupted_dataset))
