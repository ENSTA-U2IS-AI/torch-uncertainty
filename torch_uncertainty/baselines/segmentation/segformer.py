from typing import Literal

from einops import rearrange
from torch import Tensor, nn
from torchvision.transforms.v2 import functional as F

from torch_uncertainty.models.segmentation.segformer import (
    segformer_b0,
    segformer_b1,
    segformer_b2,
    segformer_b3,
    segformer_b4,
    segformer_b5,
)
from torch_uncertainty.routines.segmentation import SegmentationRoutine


class SegFormer(SegmentationRoutine):
    single = ["std"]
    versions = {
        "std": [
            segformer_b0,
            segformer_b1,
            segformer_b2,
            segformer_b3,
            segformer_b4,
            segformer_b5,
        ]
    }
    archs = [0, 1, 2, 3, 4, 5]

    def __init__(
        self,
        num_classes: int,
        loss: type[nn.Module],
        version: Literal["std"],
        arch: int,
        num_estimators: int = 1,
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

            num_estimators (int, optional): _description_. Defaults to 1.
        """
        params = {
            "num_classes": num_classes,
        }

        format_batch_fn = nn.Identity()

        if version not in self.versions:
            raise ValueError(f"Unknown version {version}")

        model = self.versions[version][self.archs.index(arch)](**params)

        super().__init__(
            num_classes=num_classes,
            model=model,
            loss=loss,
            num_estimators=num_estimators,
            format_batch_fn=format_batch_fn,
        )
        self.save_hyperparameters()

    def training_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        img, target = batch
        logits = self.forward(img)
        target = F.resize(
            target, logits.shape[-2:], interpolation=F.InterpolationMode.NEAREST
        )
        logits = rearrange(logits, "b c h w -> (b h w) c")
        target = target.flatten()
        valid_mask = target != 255
        loss = self.criterion(logits[valid_mask], target[valid_mask])
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self, batch: tuple[Tensor, Tensor], batch_idx: int
    ) -> None:
        img, target = batch
        logits = self.forward(img)
        target = F.resize(
            target, logits.shape[-2:], interpolation=F.InterpolationMode.NEAREST
        )
        logits = rearrange(logits, "b c h w -> (b h w) c")
        target = target.flatten()
        valid_mask = target != 255
        self.val_seg_metrics.update(logits[valid_mask], target[valid_mask])

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> None:
        img, target = batch
        logits = self.forward(img)
        target = F.resize(
            target, logits.shape[-2:], interpolation=F.InterpolationMode.NEAREST
        )
        logits = rearrange(logits, "b c h w -> (b h w) c")
        target = target.flatten()
        valid_mask = target != 255
        self.test_seg_metrics.update(logits[valid_mask], target[valid_mask])
