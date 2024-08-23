import os
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class TinyImageNet(Dataset):
    """Inspired by
    https://gist.github.com/z-a-f/b862013c0dc2b540cf96a123a6766e54.
    """

    def __init__(
        self,
        root: str | Path,
        split: Literal["train", "val", "test"] = "train",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ) -> None:
        self.root = Path(root) / "tiny-imagenet-200"

        if split not in ["train", "val", "test"]:
            raise ValueError(f"Split {split} is not supported.")

        self.split = split
        self.label_idx = 1
        self.transform = transform
        self.target_transform = target_transform

        self.wnids_path = self.root / "wnids.txt"
        self.words_path = self.root / "words.txt"

        self.make_dataset()

    def make_dataset(self) -> None:
        self.samples_paths = self._make_paths()
        self.samples_num = len(self.samples_paths)

        labels = []
        samples = []
        for idx in range(self.samples_num):
            s = self.samples_paths[idx]
            img = Image.open(s[0])
            img = self._add_channels(np.uint8(img))
            img = Image.fromarray(img)
            samples.append(img)
            labels.append(s[self.label_idx])

        self.samples = samples
        self.label_data = torch.as_tensor(labels).long()

    def _add_channels(self, img: np.ndarray) -> np.ndarray:
        while len(img.shape) < 3:  # third axis is the channels
            img = np.expand_dims(img, axis=-1)
        while (img.shape[-1]) < 3:
            img = np.concatenate([img, img[:, :, -1:]], axis=-1)
        return img

    def __len__(self) -> int:
        """The number of samples in the dataset."""
        return self.samples_num

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the samples and targets of the dataset.

        Args:
            index (int): The index of the sample to get.
        """
        sample = self.samples[index]
        target = self.label_data[index]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def _make_paths(self) -> list[tuple[Path, int]]:
        self.ids = []
        with self.wnids_path.open() as idf:
            for nid in idf:
                snid = nid.strip()
                self.ids.append(snid)
        self.nid_to_words = defaultdict(list)

        with self.words_path.open() as wf:
            for line in wf:
                nid, labels = line.split("\t")
                labels = [x.strip() for x in labels.split(",")]
                self.nid_to_words[nid].extend(labels)

        paths = []

        if self.split == "train":
            train_path = self.root / "train"
            train_nids = os.listdir(train_path)
            for nid in train_nids:
                anno_path = train_path / nid / (nid + "_boxes.txt")
                imgs_path = train_path / nid / "images"
                label_id = self.ids.index(nid)
                with anno_path.open() as annof:
                    for line in annof:
                        fname, _, _, _, _ = line.split()
                        fname = imgs_path / fname
                        paths.append((fname, label_id))

        elif self.split == "val":
            val_path = self.root / "val"
            with (val_path / "val_annotations.txt").open() as valf:
                for line in valf:
                    fname, nid, _, _, _, _ = line.split()
                    fname = val_path / "images" / fname
                    label_id = self.ids.index(nid)
                    paths.append((fname, label_id))

        else:  # self.split == "test":
            test_path = self.root / "test"
            paths = [test_path / x for x in os.listdir(test_path)]
        return paths
