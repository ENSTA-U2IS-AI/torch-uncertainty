from collections.abc import Callable

import torch
from torch.utils.data import Dataset
from tslearn.datasets import UCR_UEA_datasets


class UCRUEADataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
    ):
        """UCR/UEA Time Series Classification Dataset.

        Args:
            dataset_name (str): Name of the dataset to load.
            split (str, optional): Split to use (``"train"`` or ``"test"``). Defaults to ``"train"``.
            transform (Callable | None, optional): Transform to apply to the input data. Defaults
                to ``None``.
            target_transform (Callable | None, optional): Transform to apply to the target data.
                Defaults to ``None``.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        self._make_dataset()
        self.classes = torch.unique(self.data["train"][1])
        self.min_label = self.classes.min().item()

    def __len__(self) -> int:
        """Get the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.data[self.split][0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a data sample from the dataset.

        Args:
            index (int): The index of the sample to retrieve.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple containing the input tensor and the target tensor.
        """
        inputs, target = self.data[self.split][0][index], self.data[self.split][1][index]
        target = target - self.min_label  # Normalize labels to start from 0

        if self.transform:
            inputs = self.transform(inputs)

        if self.target_transform:
            target = self.target_transform(target)

        return inputs, target

    def _make_dataset(self):
        """Create the dataset from the loaded data."""
        dataloader = UCR_UEA_datasets()
        x_train, y_train, x_test, y_test = dataloader.load_dataset(self.dataset_name)
        self.data = {
            "train": [
                torch.tensor(x_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.int64),
            ],
            "test": [
                torch.tensor(x_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.int64),
            ],
        }

        self._compute_statistics()
        self._standardize()

    def _compute_statistics(self) -> None:
        self.data_mean = torch.mean(self.data["train"][0], dim=0, keepdim=True)
        self.data_std = torch.std(self.data["train"][0], dim=0, keepdim=True)

    def _standardize(self) -> torch.Tensor:
        """Standardize the input data."""
        self.data["train"][0] = (self.data["train"][0] - self.data_mean) / self.data_std
        self.data["test"][0] = (self.data["test"][0] - self.data_mean) / self.data_std
