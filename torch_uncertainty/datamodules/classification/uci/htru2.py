from pathlib import Path

from torch_uncertainty.datasets.classification import HTRU2

from .uci_classification import UCIClassificationDataModule


class HTRU2DataModule(UCIClassificationDataModule):
    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        val_split: float = 0.0,
        test_split: float = 0.2,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        binary: bool = True,
    ) -> None:
        """The HTRU2 UCI classification datamodule.

        Args:
            root (string): Root directory of the datasets.
            batch_size (int): The batch size for training and testing.
            val_split (float, optional): Share of validation samples among the
                non-test samples. Defaults to ``0``.
            test_split (float, optional): Share of test samples. Defaults to ``0.2``.
            num_workers (int, optional): How many subprocesses to use for data
                loading. Defaults to ``1``.
            pin_memory (bool, optional): Whether to pin memory in the GPU. Defaults
                to ``True``.
            persistent_workers (bool, optional): Whether to use persistent workers.
                Defaults to ``True``.
            binary (bool, optional): Whether to use binary classification. Defaults
                to ``True``.

        """
        super().__init__(
            root=root,
            dataset=HTRU2,
            batch_size=batch_size,
            val_split=val_split,
            test_split=test_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            binary=binary,
        )
