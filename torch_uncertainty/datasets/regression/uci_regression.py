import logging
from collections.abc import Callable
from importlib import util
from pathlib import Path

if util.find_spec("pandas"):
    import pandas as pd

    pandas_installed = True
else:  # coverage: ignore
    pandas_installed = False


import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.datasets.utils import (
    check_integrity,
    download_and_extract_archive,
    download_url,
)

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

energy_prediction_column_names = [
    "Appliances",
    "lights",
    "T1",
    "RH_1",
    "T2",
    "RH_2",
    "T3",
    "RH_3",
    "T4",
    "RH_4",
    "T5",
    "RH_5",
    "T6",
    "RH_6",
    "T7",
    "RH_7",
    "T8",
    "RH_8",
    "T9",
    "RH_9",
    "T_out",  # Dropped
]


class UCIRegression(Dataset):
    """The UCI regression datasets.

    Args:
        root (str): Root directory of the datasets.
        train (bool, optional): If True, creates dataset from training set,
            otherwise creates from test set.
        transform (callable, optional): A function/transform that takes in a
            numpy array and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        dataset_name (str, optional): The name of the dataset. One of
            "boston-housing", "concrete", "energy", "kin8nm",
            "naval-propulsion-plant", "power-plant", "protein",
            "wine-quality-red", and "yacht".
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
        "boston",
        "concrete",
        "energy-efficiency",
        "energy-prediction",
        "kin8nm",
        "naval-propulsion-plant",
        "power-plant",
        "protein",
        "wine-quality-red",
        "yacht",
    ]

    md5_tgz = [
        "d4accdce7a25600298819f8e28e8d593",
        "eba3e28907d4515244165b6b2c311b7b",
        "2018fb7b50778fdc1304d50a78874579",
        "d0f0f8ceaaf45df2233ce0600097bd84",
        "df08c665b7665809e74e32b107836a3a",
        "54f4febcf51bdba12e1ca63e28b3e973",
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
        "https://archive.ics.uci.edu/static/public/374/appliances+energy+"
        "prediction.zip",
        "https://www.openml.org/data/get_csv/3626/dataset_2175_kin8nm.arff",
        "https://raw.githubusercontent.com/luishpinto/cm-naval-propulsion-"
        "plant/master/data.csv",
        "https://archive.ics.uci.edu/static/public/294/combined+cycle+power+"
        "plant.zip",
        "https://archive.ics.uci.edu/static/public/265/physicochemical+"
        "properties+of+protein+tertiary+structure.zip",
        "https://archive.ics.uci.edu/static/public/186/wine+quality.zip",
        "https://archive.ics.uci.edu/static/public/243/yacht+"
        "hydrodynamics.zip",
    ]

    def __init__(
        self,
        root: Path | str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        dataset_name: str = "energy",
        download: bool = False,
        seed: int = 42,
        shuffle: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.seed = seed
        self.shuffle = shuffle

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

    def __len__(self) -> int:
        """Get the length of the dataset."""
        return self.data.shape[0]

    def _check_integrity(self) -> bool:
        """Check the integrity of the dataset(s)."""
        return check_integrity(
            self.root / self.root_appendix / Path(self.start_filename),
            self.md5,
        )

    def _standardize(self) -> None:
        self.data = (self.data - self.data_mean) / self.data_std
        self.targets = (self.targets - self.target_mean) / self.target_std

    def _compute_statistics(self) -> None:
        self.data_mean = self.data.mean(axis=0)
        self.data_std = self.data.std(axis=0)
        self.data_std[self.data_std == 0] = 1
        self.target_mean = self.targets.mean(axis=0)
        self.target_std = self.targets.std(axis=0)

    def download(self) -> None:
        """Download and extract dataset."""
        if self._check_integrity():
            logging.info("Files already downloaded and verified")
            return
        if self.url is None:
            raise ValueError(
                f"The dataset {self.dataset_name} is not available for "
                "download."
            )
        download_root = self.root / self.root_appendix / self.dataset_name
        if self.dataset_name == "boston":
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
        elif self.dataset_name == "naval-propulsion-plant":
            download_url(
                self.url,
                root=download_root,
                filename="data.csv",
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
        if not pandas_installed:  # coverage: ignore
            raise ImportError(
                "Please install torch_uncertainty with the tabular option:"
                """pip install -U "torch_uncertainty[tabular]"."""
            )
        path = self.root / self.root_appendix / self.dataset_name
        if self.dataset_name == "boston":
            array = pd.read_table(
                path / "housing.data",
                names=boston_column_names,
                header=None,
                delim_whitespace=True,
            )
        elif self.dataset_name == "concrete":
            array = pd.read_excel(path / "Concrete_Data.xls").to_numpy()
        elif self.dataset_name == "energy-efficiency":
            array = pd.read_excel(path / "ENB2012_data.xlsx").to_numpy()
        elif self.dataset_name == "energy-prediction":
            array = pd.read_csv(path / "energydata_complete.csv")[
                energy_prediction_column_names
            ].to_numpy()
        elif self.dataset_name == "kin8nm":
            array = pd.read_csv(path / "kin8nm.csv").to_numpy()
        elif self.dataset_name == "naval-propulsion-plant":
            df = pd.read_csv(
                path / "data.csv", header=None, sep=";", decimal=","
            )
            # convert Ex to 10^x and remove second target
            array = df.apply(pd.to_numeric, errors="coerce").to_numpy()[:, :-1]
        elif self.dataset_name == "protein":
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

        array = torch.as_tensor(array).float()

        if self.dataset_name == "energy-efficiency":
            self.data = array[:, 2:-3]
            self.targets = array[:, -2]
        else:
            self.data = array[:, :-1]
            self.targets = array[:, -1]

        self._compute_statistics()
        self._standardize()

        if self.dataset_name == "energy-prediction":
            self.data = F.pad(self.data, (0, 0, 13, 0), value=0)

        if self.shuffle:
            gen = torch.Generator()
            gen.manual_seed(self.seed)
            indexes = torch.randperm(array.shape[0], generator=gen)
            array = array[indexes]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get sample and target for a given index."""
        if self.dataset_name == "energy-prediction":
            data = self.data[index : index + 13, :]
            target = self.data[index : index + 13, :]
            return data, target

        data = self.data[index]
        if self.transform is not None:
            data = self.transform(data)
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target
