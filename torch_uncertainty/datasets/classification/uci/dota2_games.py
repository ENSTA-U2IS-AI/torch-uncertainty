import logging
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torchvision.datasets.utils import download_and_extract_archive

from .uci_classification import UCIClassificationDataset


class DOTA2Games(UCIClassificationDataset):
    """The DOTA 2 Games UCI classification dataset.

    Args:
        root (str): Root directory of the datasets.
        train (bool, optional): If True, creates dataset from training set,
            otherwise creates from test set.
        transform (callable, optional): A function/transform that takes in a
            numpy array and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again.
        binary (bool, optional): Whether to use binary classification. Defaults
            to ``True``.

    Note - License:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.

    """

    md5_zip = "896623c082b062f56b9c49c6c1fc0bf7"
    url = (
        "https://archive.ics.uci.edu/static/public/367/dota2+games+results.zip"
    )
    dataset_name = "dota2+games+results"
    filename = "dota2Train.csv"
    num_features = 116
    need_split = False

    def __init__(
        self,
        root: Path | str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        binary: bool = True,
        download: bool = False,
        train: bool = True,
        test_split: float = 0.2,
        split_seed: int = 21893027,
    ) -> None:
        super().__init__(
            root,
            transform,
            target_transform,
            binary,
            download,
            train,
            test_split,
            split_seed,
        )

    def download(self) -> None:
        """Download and extract dataset."""
        if self._check_integrity():
            logging.info("Files already downloaded and verified")
            return
        download_and_extract_archive(
            self.url,
            download_root=self.root,
            extract_root=self.root / "dota2_games",
            filename="dota2+games+results.zip",
            md5=self.md5_zip,
        )

    def _make_dataset(self) -> None:
        """Create dataset from extracted files."""
        path = (
            self.root
            / "dota2_games"
            / ("dota2Train.csv" if self.train else "dota2Test.csv")
        )

        data = pd.read_csv(path, sep=",", header=None)
        data[0] = np.where(data[0] == 1, 1, 0)
        self.targets = torch.as_tensor(data[0].values, dtype=torch.long)

        self.data = data.drop(columns=[0])
        categorical_columns = self.data.select_dtypes(include="object").columns
        for col in categorical_columns:
            if self.data[col].nunique() == 2:
                self.data[col] = np.where(self.data[col] == "yes", 1, 0)
        self.data = torch.as_tensor(
            pd.get_dummies(self.data).astype(float).values, dtype=torch.float32
        )
        self.num_features = self.data.shape[1]
