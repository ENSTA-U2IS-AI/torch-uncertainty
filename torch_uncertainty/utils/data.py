import copy
from collections.abc import Callable
from typing import Any

from torch.utils.data import Dataset, random_split


def create_train_val_split(
    dataset: Dataset,
    val_split_rate: float,
    val_transforms: Callable | None = None,
) -> tuple[Dataset, Dataset]:
    train, val = random_split(dataset, [1 - val_split_rate, val_split_rate])
    val = copy.deepcopy(val)
    val.dataset.transform = val_transforms
    return train, val


class TTADataset(Dataset):
    def __init__(self, dataset: Dataset, num_augmentations: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.num_augmentations = num_augmentations

    def __len__(self):
        """Get the virtual length of the dataset."""
        return len(self.dataset) * self.num_augmentations

    def __getitem__(self, index) -> Any:
        """Get the item corresponding to idx // :attr:`self.num_augmentations`."""
        return self.dataset[index // self.num_augmentations]
