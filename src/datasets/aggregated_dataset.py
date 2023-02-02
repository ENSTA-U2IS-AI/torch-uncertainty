import torch
from torch.utils.data import Dataset


class AggregatedDataset(Dataset):
    def __init__(self, dataset: Dataset, n_dataloaders: int) -> None:
        super().__init__()
        self.dataset = dataset
        self.n_dataloaders = n_dataloaders
        self.dataset_size = len(self.dataset)
        self.offset = self.dataset_size // self.n_dataloaders

    def __getitem__(self, idx: int):
        inputs, targets = zip(
            *[
                self.dataset[(idx + i * self.offset) % self.dataset_size]
                for i in range(self.n_dataloaders)
            ]
        )
        inputs = torch.cat(inputs, dim=0)
        targets = torch.as_tensor(targets)
        return inputs, targets

    def __len__(self):
        return self.dataset_size
