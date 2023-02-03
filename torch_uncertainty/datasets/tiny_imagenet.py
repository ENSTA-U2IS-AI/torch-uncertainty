# fmt:off
import os
from collections import defaultdict
from typing import Callable, Optional

from PIL import Image
from torch.utils.data import Dataset

import numpy as np


# fmt:on
class TinyImageNetPaths:
    """From https://gist.github.com/z-a-f/b862013c0dc2b540cf96a123a6766e54."""

    def __init__(self, root):
        train_path = root / "train"
        val_path = root / "val"
        test_path = root / "test"

        wnids_path = root / "wnids.txt"
        words_path = root / "words.txt"

        self._make_paths(
            train_path, val_path, test_path, wnids_path, words_path
        )

    def _make_paths(
        self, train_path, val_path, test_path, wnids_path, words_path
    ):
        self.ids = []
        with open(wnids_path, "r") as idf:
            for nid in idf:
                nid = nid.strip()
                self.ids.append(nid)
        self.nid_to_words = defaultdict(list)
        with open(words_path, "r") as wf:
            for line in wf:
                nid, labels = line.split("\t")
                labels = list(map(lambda x: x.strip(), labels.split(",")))
                self.nid_to_words[nid].extend(labels)

        self.paths = {
            "train": [],  # [img_path, id, nid, box]
            "val": [],  # [img_path, id, nid, box]
            "test": [],  # img_path
        }

        # Get the test paths
        self.paths["test"] = list(
            map(lambda x: os.path.join(test_path, x), os.listdir(test_path))
        )
        # Get the validation paths and labels
        with open(os.path.join(val_path, "val_annotations.txt")) as valf:
            for line in valf:
                fname, nid, x0, y0, x1, y1 = line.split()
                fname = os.path.join(val_path, "images", fname)
                bbox = int(x0), int(y0), int(x1), int(y1)
                label_id = self.ids.index(nid)
                self.paths["val"].append((fname, label_id, nid, bbox))

        # Get the training paths
        train_nids = os.listdir(train_path)
        for nid in train_nids:
            anno_path = os.path.join(train_path, nid, nid + "_boxes.txt")
            imgs_path = os.path.join(train_path, nid, "images")
            label_id = self.ids.index(nid)
            with open(anno_path, "r") as annof:
                for line in annof:
                    fname, x0, y0, x1, y1 = line.split()
                    fname = os.path.join(imgs_path, fname)
                    bbox = int(x0), int(y0), int(x1), int(y1)
                    self.paths["train"].append((fname, label_id, nid, bbox))


class TinyImageNet(Dataset):
    """From https://gist.github.com/z-a-f/b862013c0dc2b540cf96a123a6766e54."""

    def __init__(
        self,
        root: str,
        split="train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        max_samples=None,
    ):
        self.split = split
        self.label_idx = 1  # from [image, id, nid, box]
        self.transform = transform
        self.target_transform = target_transform
        self.root = root

        self.IMAGE_SHAPE = (64, 64, 3)

        self.samples = []
        self.label_data = []

        self.max_samples = max_samples
        self.samples_num = len(self.samples)

        if self.max_samples is not None:
            self.samples_num = min(self.max_samples, self.samples_num)
            self.samples = np.random.permutation(self.samples)[
                : self.samples_num
            ]

        self.samples = np.zeros(
            (self.samples_num,) + self.IMAGE_SHAPE, dtype=np.float32
        )

        self.make_dataset(self.root)

    def make_dataset(self, directory: str):
        tinp = TinyImageNetPaths(directory)
        self.samples = tinp.paths[self.split]
        self.samples_num = len(self.samples)

        self.label_data = np.zeros((self.samples_num,), dtype=np.int)
        for idx in range(self.samples_num):
            s = self.samples[idx]
            # TODO: check that it works
            img = Image.open(s[0])
            # img = mpimg.imread(s[0])
            img = self._add_channels(img)
            img = Image.fromarray(np.uint8(img))
            self.samples[idx] = img
            if self.split != "test":
                self.label_data[idx] = s[self.label_idx]

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
        target = None if self.split == "test" else self.label_data[index]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
