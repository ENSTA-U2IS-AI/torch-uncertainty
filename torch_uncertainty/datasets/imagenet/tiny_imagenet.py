# fmt:off
import os
from collections import defaultdict
from pathlib import Path
from typing import Callable, Literal, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

import numpy as np


# fmt:on
class TinyImageNet(Dataset):
    """Inspired by
    https://gist.github.com/z-a-f/b862013c0dc2b540cf96a123a6766e54.
    """

    def __init__(
        self,
        root: str,
        split: Literal["train", "val", "test"] = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        self.root = Path(root) / "tiny-imagenet-200"

        self.split = split
        self.label_idx = 1  # from [image, id, nid, box]
        self.transform = transform
        self.target_transform = target_transform

        self.wnids_path = self.root / "wnids.txt"
        self.words_path = self.root / "words.txt"

        self.make_dataset(self.root)

    def make_dataset(self, directory: str):
        self.samples = self._make_paths()
        self.samples_num = len(self.samples)

        labels = []
        for idx in range(self.samples_num):
            s = self.samples[idx]
            img = Image.open(s[0])
            img = self._add_channels(np.uint8(img))
            img = Image.fromarray(img)
            self.samples[idx] = img
            labels.append(s[self.label_idx])

        self.samples = self.samples
        self.label_data = torch.as_tensor(labels).long()

    def _add_channels(self, img):
        while len(img.shape) < 3:  # third axis is the channels
            img = np.expand_dims(img, axis=-1)
        while (img.shape[-1]) < 3:
            img = np.concatenate([img, img[:, :, -1:]], axis=-1)
        return img

    def __len__(self):
        return self.samples_num

    def __getitem__(self, index):
        sample = self.samples[index]
        target = self.label_data[index]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def _make_paths(self):
        self.ids = []
        with open(self.wnids_path, "r") as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)

        with open(self.words_path, "r") as wf:
            for line in wf:
                nid, labels = line.split("\t")
                labels = list(map(lambda x: x.strip(), labels.split(",")))
                self.nid_to_words[nid].extend(labels)

        paths = []

        if self.split == "train":
            train_path = self.root / "train"
            train_nids = os.listdir(train_path)
            for nid in train_nids:
                anno_path = os.path.join(train_path, nid, nid + "_boxes.txt")
                imgs_path = os.path.join(train_path, nid, "images")
                label_id = self.ids.index(nid)
                with open(anno_path, "r") as annof:
                    for line in annof:
                        fname, x0, y0, x1, y1 = line.split()
                        fname = os.path.join(imgs_path, fname)
                        paths.append((fname, label_id))

        elif self.split == "val":
            val_path = self.root / "val"
            with open(os.path.join(val_path, "val_annotations.txt")) as valf:
                for line in valf:
                    fname, nid, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(val_path, "images", fname)
                    label_id = self.ids.index(nid)
                    paths.append((fname, label_id))

        else:  # self.split == "test":
            test_path = self.root / "test"
            paths = list(
                map(lambda x: os.path.join(test_path, x), os.listdir(test_path))
            )
        return paths
