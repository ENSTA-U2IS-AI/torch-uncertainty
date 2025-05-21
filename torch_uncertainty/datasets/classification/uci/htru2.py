from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch

from .uci_classification import UCIClassificationDataset


class HTRU2(UCIClassificationDataset):
    md5_zip = "1cfbf71c604debc06dedcbb6c1ccb43f"
    url = "https://archive.ics.uci.edu/static/public/372/htru2.zip"
    dataset_name = "htru2"
    filename = "HTRU_2.csv"
    num_features = 8

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
        """The HTRU2 UCI classification dataset.

        Args:
            root (str | Path): Root directory of the datasets.
            train (bool, optional): If ``True``, creates dataset from training set,
                otherwise creates from test set.
            transform (callable, optional): A function/transform that takes in a
                numpy array and returns a transformed version. Defaults to ``None``.
            target_transform (callable, optional): A function/transform that takes
                in the target and transforms it. Defaults to ``None``.
            download (bool, optional): If ``True``, downloads the dataset from the
                internet and puts it in root directory. If dataset is already
                downloaded, it is not downloaded again. Defaults to ``False``.
            binary (bool, optional): Whether to use binary classification. Defaults
                to ``True``.
            test_split (float, optional): The fraction of the dataset to use as test set.
                Defaults to ``0.2``.
            split_seed (int, optional): The random seed for splitting the dataset.
                Defaults to ``21893027``.

        Note:
            The licenses of the datasets may differ from TorchUncertainty's
            license. Check before use.
        """
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

    def _make_dataset(self) -> None:
        """Create dataset from extracted files."""
        data = pd.read_csv(self.root / self.dataset_name / self.filename, sep=",", header=None)
        self.targets = torch.as_tensor(data[8].values, dtype=torch.long)
        self.data = torch.as_tensor(data.drop(columns=[8]).values, dtype=torch.float32)
        self.num_features = self.data.shape[1]
