from typing import Literal

from torch import nn

from torch_uncertainty.models.segmentation.segformer import (
    seg_former,
)
from torch_uncertainty.routines.segmentation import SegmentationRoutine


class SegFormerBaseline(SegmentationRoutine):
    single = ["std"]
    versions = {
        "std": seg_former,
    }
    archs = [0, 1, 2, 3, 4, 5]

    def __init__(
        self,
        num_classes: int,
        loss: nn.Module,
        version: Literal["std"],
        arch: int,
    ) -> None:
        r"""SegFormer backbone baseline for segmentation providing support for
        various versions and architectures.

        Args:
            num_classes (int): Number of classes to predict.
            loss (type[Module]): Training loss.
            version (str):
                Determines which SegFormer version to use. Options are:

                - ``"std"``: original SegFormer

            arch (int):
                Determines which architecture to use. Options are:

                - ``0``: SegFormer-B0
                - ``1``: SegFormer-B1
                - ``2``: SegFormer-B2
                - ``3``: SegFormer-B3
                - ``4``: SegFormer-B4
                - ``5``: SegFormer-B5

            num_estimators (int, optional): Number of estimators in the
                ensemble. Defaults to 1 (single model).
        """
        params = {
            "num_classes": num_classes,
            "arch": arch,
        }

        format_batch_fn = nn.Identity()

        if version not in self.versions:
            raise ValueError(f"Unknown version {version}")

        model = self.versions[version](**params)

        super().__init__(
            num_classes=num_classes,
            model=model,
            loss=loss,
            format_batch_fn=format_batch_fn,
        )
        self.save_hyperparameters(ignore=["loss"])
