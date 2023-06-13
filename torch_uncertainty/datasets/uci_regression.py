# fmt: off
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    download_url,
)


# fmt:on
class UCIRegression(Dataset):
    """The UCI regression datasets.

    Args:
        root (string): Root directory of the datasets.
        train (bool, optional): If True, creates dataset from training set,
            otherwise creates from test set.
        transform (callable, optional): A function/transform that takes in a
            numpy array and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        dataset_name (string, optional): The name of the dataset. One of
            "boston-housing", "concrete", "energy", "kin8nm",
            "naval-propulsion-plant", "power-plant",
            "protein-tertiary-structure", "wine-quality-red", "yacht".
        download (bool, optional): If true, downloads the dataset from the
            internet and puts it in root directory. If dataset is already
            downloaded, it is not downloaded again.

    Note - Ethics:
        You may want to avoid using the boston-housing dataset because of
        ethical concerns.

    Note - License:
        The licenses of the datasets may differ from TorchUncertainty's
        license. Check before use.
    """

    root_appendix = "uci_regression"
    uci_subsets = [
        "boston-housing",
        "concrete",
        "energy",
        "kin8nm",
        "naval-propulsion-plant",
        "power-plant",
        "protein-tertiary-structure",
        "wine-quality-red",
        "yacht",
    ]

    md5_tgz = [
        "d4accdce7a25600298819f8e28e8d593",
        "eba3e28907d4515244165b6b2c311b7b",
        "2018fb7b50778fdc1304d50a78874579",
        "df08c665b7665809e74e32b107836a3a",
        None,
        "f5065a616eae05eb4ecae445ecf6e720",
        "37bcb77a8abad274a987439e6a3de632",
        "0ddfa7a9379510fe7ff88b9930e3c332",
        "4e6727f462779e2d396e8f7d2ddb79a3",
    ]
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/"
        "housing.data",
        "https://archive.ics.uci.edu/static/public/165/concrete+compressive+"
        "strength.zip",
        "https://archive.ics.uci.edu/static/public/242/energy+efficiency.zip",
        "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.arff",
        "https://archive.ics.uci.edu/static/public/316/condition+based+"
        "maintenance+of+naval+propulsion+plants.zip",
        "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+"
        "plant.zip",
        "https://archive.ics.uci.edu/static/public/265/physicochemical+"
        "properties+of+protein+tertiary+structure.zip",
        "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
        "https://archive.ics.uci.edu/static/public/243/yacht+"
        "hydrodynamics.zip",
    ]

    boston_column_names = [
        "CRIM",
        "ZN",
        "INDUS",
        "CHAS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "RAD",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
        "MEDV",
    ]

    def __init__(
        self,
        root: Union[Path, str],
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        dataset_name: str = "energy",
        download: bool = False,
    ) -> None:
        super().__init__()

        if isinstance(root, str):
            root = Path(root)
        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # self.standardize = standardize

        if dataset_name not in self.uci_subsets:
            raise ValueError(
                f"The dataset {dataset_name} is not implemented. "
                "`dataset_name` should be one of {self.uci_subsets}."
            )
        self.dataset_name = dataset_name
        dataset_id = self.uci_subsets.index(dataset_name)
        self.url = self.urls[dataset_id]
        self.start_filename = self.url.split("/")[-1]
        self.md5 = self.md5_tgz[dataset_id]

        if download:
            self.download()

        self._make_dataset()

        self._standardize()

    def __len__(self) -> int:
        return self.data.shape[0]

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset(s)."""
        return check_integrity(
            self.root / self.root_appendix / Path(self.start_filename),
            self.md5,
        )

    def _standardize(self):
        self.data_mean = self.data.mean(axis=0)
        self.data_std = self.data.std(axis=0)
        self.data_std[self.data_std == 0] = 1

        self.target_mean = self.targets.mean(axis=0)
        self.target_std = self.targets.std(axis=0)

        self.data = (self.data - self.data_mean) / self.data_std
        self.targets = (self.targets - self.target_mean) / self.target_std

    def download(self) -> None:
        """Download and extract dataset."""
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        if self.url is None:
            raise ValueError(
                f"The dataset {self.dataset_name} is not available for "
                "download."
            )
        download_root = self.root / self.root_appendix / self.dataset_name
        if self.dataset_name == "boston-housing":
            download_url(
                self.url,
                root=download_root,
                filename="housing.data",
            )
        elif self.dataset_name == "kin8nm":
            download_url(
                self.url,
                root=download_root,
                filename="kin8nm.csv",
            )
        else:
            download_and_extract_archive(
                self.url,
                download_root=download_root,
                extract_root=download_root,
                filename=self.start_filename,
                md5=self.md5,
            )

    def _make_dataset(self) -> None:
        """Create dataset from extracted files."""
        path = self.root / self.root_appendix / self.dataset_name
        if self.dataset_name == "boston-housing":
            array = pd.read_table(
                path / "housing.data",
                names=self.boston_column_names,
                header=None,
                delim_whitespace=True,
            )
        elif self.dataset_name == "concrete":
            array = pd.read_excel(path / "Concrete_Data.xls").to_numpy()
        elif self.dataset_name == "energy":
            array = pd.read_excel(path / "ENB2012_data.xlsx").to_numpy()
        elif self.dataset_name == "kin8nm":
            array = pd.read_csv(path / "kin8nm.csv").to_numpy()
        # elif self.dataset_name == "naval-propulsion-plant":
        #     array = pd.read_csv(
        #         path / "naval-propulsion-plant.csv",
        #         header=None,
        #     ).to_numpy()
        # elif self.dataset_name == "power-plant":
        #     array = pd.read_excel(path / "Folds5x2_pp.xlsx").to_numpy()
        elif self.dataset_name == "protein-tertiary-structure":
            array = pd.read_csv(
                path / "CASP.csv",
            ).to_numpy()
        elif self.dataset_name == "wine-quality-red":
            array = pd.read_csv(
                path / "winequality-red.csv",
                sep=";",
            ).to_numpy()
        elif self.dataset_name == "yacht":
            array = pd.read_csv(
                path / "yacht_hydrodynamics.data",
                delim_whitespace=True,
                header=None,
            ).to_numpy()
        else:
            raise ValueError("Dataset not implemented.")
        self.data = torch.as_tensor(array[:, :-1]).float()
        self.targets = torch.as_tensor(array[:, -1]).float()

        if self.train:
            self.data = self.data[: int(0.8 * len(self.data))]
            self.targets = self.targets[: int(0.8 * len(self.targets))]
        else:
            self.data = self.data[int(0.8 * len(self.data)) :]
            self.targets = self.targets[int(0.8 * len(self.targets)) :]

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sample and target for a given index."""
        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target