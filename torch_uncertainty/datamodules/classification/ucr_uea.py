from torch import Generator
from torch.utils.data import DataLoader, random_split

from torch_uncertainty.datamodules.abstract import TUDataModule
from torch_uncertainty.datasets.classification.ucr_uea import UCRUEADataset


class UCRUEADataModule(TUDataModule):
    """Data module for UCR/UEA Time Series Classification datasets."""

    def __init__(
        self,
        dataset_name: str,
        batch_size: int,
        eval_batch_size: int | None = None,
        val_split: float | None = None,
        eval_ood: bool = False,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        split_seed: int = 42,
    ) -> None:
        """Initialize the UCR/UEA dataset.

        Args:
            dataset_name (str): Name of the dataset to load.
            batch_size (int): The batch size for training and testing.
            eval_batch_size (int | None): Number of samples per batch during evaluation (val and
                test). Set to :attr:`batch_size` if ``None``. Defaults to ``None``.
            val_split (float | None): Share of validation samples. Defaults to ``0``.
            eval_ood (bool): Whether to evaluate on out-of-distribution (OOD) data. Defaults to
                ``False``.
            num_workers (int, optional): How many subprocesses to use for data loading. Defaults
                to ``1``.
            pin_memory (bool, optional): Whether to pin memory in the GPU. Defaults to ``True``.
            persistent_workers (bool, optional): Whether to use persistent workers. Defaults to ``True``.
            split_seed (int, optional): The seed to use for splitting the dataset.
                Defaults to ``42``.
        """
        super().__init__(
            root=f"~/.tslearn/datasets/UCR_UEA/{dataset_name}",
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            val_split=val_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        self.dataset_name = dataset_name
        self.eval_ood = eval_ood
        self.gen = Generator().manual_seed(split_seed)

    def prepare_data(self):
        """Download the dataset if it does not exist."""
        self.dataset = UCRUEADataset(self.dataset_name)

    def setup(self, stage: str | None = None):
        """Setup the dataset for training, validation, and testing."""
        if stage == "fit" or stage is None:
            full_dataset = UCRUEADataset(self.dataset_name, split="train", create_ood=True)

            if self.val_split is not None:
                self.train, self.val = random_split(
                    full_dataset, [1 - self.val_split, self.val_split], generator=self.gen
                )
            else:
                self.train = full_dataset
                self.val = UCRUEADataset(self.dataset_name, split="test", create_ood=True)

        if stage == "test" or stage is None:
            self.test = UCRUEADataset(self.dataset_name, split="test", create_ood=True)
            if self.eval_ood:
                self.ood = UCRUEADataset(self.dataset_name, split="ood", create_ood=True)

    def test_dataloader(self) -> list[DataLoader]:
        r"""Get test dataloaders.

        Return:
            list[DataLoader]: test set for in distribution data, and eventually out-of-distribution data.
        """
        dataloaders = [self._data_loader(self.get_test_set(), training=False, shuffle=False)]
        if self.eval_ood:
            dataloaders.append(self._data_loader(self.get_ood_set(), training=False, shuffle=False))
        return dataloaders
