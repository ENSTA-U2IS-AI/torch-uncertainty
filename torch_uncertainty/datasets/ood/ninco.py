import os
import tarfile
import urllib.request
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class NINCO(Dataset):
    """
    NINCO dataset adapted to your code base.
    
    The dataset consists of 5,879 manually verified out-of-distribution (OOD)
    images organized into 64 classes. All samples are assigned the label -1.
    
    The dataset will be downloaded from Zenodo if not available locally.
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        """
        Args:
            root (str | Path): The root directory where the dataset will be saved.
            transform (Callable, optional): A function/transform to apply to the images.
            target_transform (Callable, optional): A function/transform to apply to the labels.
            download (bool, optional): Whether to download the dataset if not present.
        """
        self.download_url = "https://zenodo.org/record/8013288/files/NINCO_all.tar.gz"
        self.filename = "NINCO_all.tar.gz"
        self.root = Path(root) / "NINCO"
        self.transform = transform
        self.target_transform = target_transform

        # Download and extract the dataset if required.
        if download and not self.root.exists():
            archive_path = Path(root) / self.filename
            print("Downloading NINCO dataset...")
            urllib.request.urlretrieve(self.download_url, archive_path)
            print("Extracting dataset...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=Path(root))
            os.remove(archive_path)

        # Ensure that the expected dataset directory exists.
        if not self.root.exists():
            raise RuntimeError("Dataset not found. You can use download=True to download it.")

        # Prepare the samples and labels.
        self.make_dataset()

    def make_dataset(self) -> None:
        """
        Loads the dataset file paths, reads each image into memory,
        and stores the image data alongside a default label (-1) for each sample.
        """
        self.samples_paths = self._make_paths()
        self.samples_num = len(self.samples_paths)

        self.samples = []
        labels = []
        for (img_path, label) in self.samples_paths:
            try:
                img = Image.open(img_path).convert("RGB")  # force RGB for consistency
            except Exception as e:
                print(f"Warning: Could not open image {img_path}. Error: {e}")
                continue
            img = np.array(img)
            img = self._add_channels(img)
            img = Image.fromarray(img)
            self.samples.append(img)
            labels.append(label)
        self.label_data = torch.as_tensor(labels).long()

    def _make_paths(self) -> list[tuple[Path, int]]:
        """
        Walks through the dataset folders to create a list of tuples.
        Each tuple contains the image path and its associated label (-1).
        """
        base_folders = ["NINCO_OOD_classes"]
        paths = []
        for folder in base_folders:
            folder_path = self.root / folder
            if not folder_path.exists():
                continue
            # Each subfolder corresponds to one OOD class.
            for class_dir in sorted(os.listdir(folder_path)):
                class_folder = folder_path / class_dir
                if os.path.isdir(class_folder):
                    for fname in sorted(os.listdir(class_folder)):
                        file_path = class_folder / fname
                        if file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                            paths.append((file_path, -1))
        return paths

    def _add_channels(self, img: np.ndarray) -> np.ndarray:
        """
        Ensures that the image has three channels.
        If the image is grayscale (i.e. has a single channel),
        extra channels are appended by duplicating the existing ones.
        """
        while len(img.shape) < 3:
            img = np.expand_dims(img, axis=-1)
        while img.shape[-1] < 3:
            img = np.concatenate([img, img[:, :, -1:]], axis=-1)
        return img

    def __len__(self) -> int:
        """Return the total number of samples in the dataset."""
        return self.samples_num

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a sample and its corresponding target (label).
        
        Args:
            index (int): The index of the sample to retrieve.
        
        Returns:
            tuple: (sample, target) where sample is a transformed image and
                   target is the label.
        """
        sample = self.samples[index]
        target = self.label_data[index]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
