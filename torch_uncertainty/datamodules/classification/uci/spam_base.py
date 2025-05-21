from pathlib import Path

from torch_uncertainty.datasets.classification import SpamBase

from .uci_classification import UCIClassificationDataModule


class SpamBaseDataModule(UCIClassificationDataModule):
    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        eval_batch_size: int | None = None,
        val_split: float = 0.0,
        test_split: float = 0.2,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        binary: bool = True,
    ) -> None:
        """The Bank Marketing UCI classification datamodule.

        Args:
            root (str | Path): Root directory of the datasets.
            batch_size (int): The batch size for training and testing.
            eval_batch_size (int | None) : Number of samples per batch during evaluation (val
                and test). Set to :attr:`batch_size` if ``None``. Defaults to ``None``.
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
            dataset=SpamBase,
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            val_split=val_split,
            test_split=test_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            binary=binary,
        )
