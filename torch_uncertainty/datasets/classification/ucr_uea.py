import warnings
from collections.abc import Callable

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import Dataset
from tslearn.datasets import UCR_UEA_datasets


class UCRUEADataset(Dataset):
    def __init__(
        self,
        dataset_name: str,
        split: str = "train",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        create_ood: bool = False,
    ):
        """UCR/UEA Time Series Classification Dataset.

        Args:
            dataset_name (str): Name of the dataset to load.
            split (str, optional): Split to use (``"train"``, ``"test"`` or ``"ood"``). Defaults to ``"train"``.
            transform (Callable | None, optional): Transform to apply to the input data. Defaults
                to ``None``.
            target_transform (Callable | None, optional): Transform to apply to the target data.
                Defaults to ``None``.
            create_ood (bool, optional): Whether to create an out-of-distribution (OOD) dataset based
                on the last class in the dataset. Defaults to ``False``.

        Raises:
            ValueError: If `split` is set to "ood" but `create_ood` is not set to True.
        """
        if split == "ood" and not create_ood:
            raise ValueError(
                "Cannot use split 'ood' without setting create_ood to True. "
                "Set create_ood=True to create an OOD dataset."
            )

        super().__init__()
        self.dataset_name = dataset_name
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.create_ood = create_ood

        if create_ood:
            warnings.warn(
                "When `create_ood` is set to True, the dataset will be created with the last class as OOD.",
                UserWarning,
                stacklevel=2,
            )

        self._make_dataset()

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
        inputs = rearrange(inputs, "t c -> c t")  # Change shape to (channels, time)

        if self.transform:
            inputs = self.transform(inputs)

        if self.target_transform:
            target = self.target_transform(target)

        return inputs, target

    def _make_dataset(self):
        """Create the dataset from the loaded data."""
        dataloader = UCR_UEA_datasets()
        x_train, y_train, x_test, y_test = dataloader.load_dataset(self.dataset_name)

        self.classes = np.unique(y_train)
        self.min_label = np.min(self.classes)

        y_train = y_train - self.min_label  # Normalize labels to start from 0
        y_test = y_test - self.min_label  # Normalize labels to start from 0

        ood_data = None

        if self.create_ood:
            train_ood_mask = y_train == (len(self.classes) - 1)
            test_ood_mask = y_test == (len(self.classes) - 1)

            x_train_ood = x_train[train_ood_mask]
            y_train_ood = y_train[train_ood_mask]
            x_test_ood = x_test[test_ood_mask]
            y_test_ood = y_test[test_ood_mask]
            x_ood = np.concatenate((x_train_ood, x_test_ood), axis=0)
            y_ood = np.concatenate((y_train_ood, y_test_ood), axis=0)

            x_train = x_train[~train_ood_mask]
            y_train = y_train[~train_ood_mask]
            x_test = x_test[~test_ood_mask]
            y_test = y_test[~test_ood_mask]

            ood_data = [
                torch.tensor(x_ood, dtype=torch.float32),
                torch.tensor(y_ood, dtype=torch.int64),
            ]

        self.data = {
            "train": [
                torch.tensor(x_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.int64),
            ],
            "test": [
                torch.tensor(x_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.int64),
            ],
            "ood": ood_data,
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
        if self.data["ood"] is not None:
            self.data["ood"][0] = (self.data["ood"][0] - self.data_mean) / self.data_std
