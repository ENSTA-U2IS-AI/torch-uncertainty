from pathlib import Path

from torchvision.transforms import v2

from torch_uncertainty.datamodules.abstract import AbstractDataModule
from torch_uncertainty.datasets.segmentation import CamVid


class CamVidDataModule(AbstractDataModule):
    def __init__(
        self,
        root: str | Path,
        batch_size: int,
        val_split: float = 0.0,
        num_workers: int = 1,
        pin_memory: bool = True,
        persistent_workers: bool = True,
    ) -> None:
        super().__init__(
            root=root,
            batch_size=batch_size,
            val_split=val_split,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        self.dataset = CamVid

        self.transform_train = v2.Compose(
            [v2.Resize((360, 480), interpolation=v2.InterpolationMode.NEAREST)]
        )
        self.transform_test = v2.Compose(
            [v2.Resize((360, 480), interpolation=v2.InterpolationMode.NEAREST)]
        )

    def prepare_data(self) -> None:  # coverage: ignore
        self.dataset(root=self.root, download=True)

    def setup(self, stage: str | None = None) -> None:
        if stage == "fit" or stage is None:
            self.train = self.dataset(
                root=self.root,
                split="train",
                download=False,
                transform=self.transform_train,
            )
            self.val = self.dataset(
                root=self.root,
                split="val",
                download=False,
                transform=self.transform_test,
            )
        elif stage == "test":
            self.test = self.dataset(
                root=self.root,
                split="test",
                download=False,
                transform=self.transform_test,
            )
        else:
            raise ValueError(f"Stage {stage} is not supported.")
