from typing import Literal

from torch import nn

from torch_uncertainty.models.segmentation.deeplab import (
    deep_lab_v3_resnet50,
    deep_lab_v3_resnet101,
)
from torch_uncertainty.routines.segmentation import SegmentationRoutine


class DeepLabBaseline(SegmentationRoutine):
    single = ["std"]
    versions = {
        "std": [
            deep_lab_v3_resnet50,
            deep_lab_v3_resnet101,
        ]
    }
    archs = [50, 101]

    def __init__(
        self,
        num_classes: int,
        loss: nn.Module,
        version: Literal["std"],
        arch: int,
        style: Literal["v3", "v3+"],
        output_stride: int,
        separable: bool,
        metric_subsampling_rate: float = 1e-2,
        log_plots: bool = False,
        num_calibration_bins: int = 15,
        pretrained_backbone: bool = True,
    ) -> None:
        params = {
            "num_classes": num_classes,
            "style": style,
            "output_stride": output_stride,
            "separable": separable,
            "pretrained_backbone": pretrained_backbone,
        }

        format_batch_fn = nn.Identity()

        if version not in self.versions:
            raise ValueError(f"Unknown version {version}")

        model = self.versions[version][self.archs.index(arch)](**params)
        super().__init__(
            num_classes=num_classes,
            model=model,
            loss=loss,
            format_batch_fn=format_batch_fn,
            metric_subsampling_rate=metric_subsampling_rate,
            log_plots=log_plots,
            num_calibration_bins=num_calibration_bins,
        )
        self.save_hyperparameters(ignore=["loss"])
