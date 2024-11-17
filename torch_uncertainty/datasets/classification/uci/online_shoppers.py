from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch

from .uci_classification import UCIClassificationDataset


class OnlineShoppers(UCIClassificationDataset):
    """The Online Shoppers Intention UCI classification dataset.

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

    md5_zip = "d835049e5f428f3b8cb8a6e6937f5537"
    url = "https://archive.ics.uci.edu/static/public/468/online+shoppers+purchasing+intention+dataset.zip"
    dataset_name = "online_shoppers"
    filename = "online_shoppers_intention.csv"
    num_features = 28

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

    def _make_dataset(self) -> None:
        """Create dataset from extracted files."""
        data = pd.read_csv(
            self.root / self.dataset_name / self.filename,
            sep=",",
            true_values=["TRUE"],
            false_values=["FALSE"],
        )
        self.targets = torch.as_tensor(data["Revenue"].values, dtype=torch.long)

        data = pd.get_dummies(data).astype(float)
        data = data.drop(columns=["Revenue"])
        self.data = torch.as_tensor(data.values, dtype=torch.float32)
        self.num_features = self.data.shape[1]
