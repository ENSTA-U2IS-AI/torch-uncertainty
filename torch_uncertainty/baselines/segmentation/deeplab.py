from typing import Literal

from torch import nn

from torch_uncertainty.models.segmentation import deep_lab_v3_resnet
from torch_uncertainty.routines.segmentation import SegmentationRoutine


class DeepLabBaseline(SegmentationRoutine):
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
        num_bins_cal_err: int = 15,
        pretrained_backbone: bool = True,
    ) -> None:
        params = {
            "num_classes": num_classes,
            "arch": arch,
            "style": style,
            "output_stride": output_stride,
            "separable": separable,
            "pretrained_backbone": pretrained_backbone,
        }

        format_batch_fn = nn.Identity()

        model = deep_lab_v3_resnet(**params)
        super().__init__(
            num_classes=num_classes,
            model=model,
            loss=loss,
            format_batch_fn=format_batch_fn,
            metric_subsampling_rate=metric_subsampling_rate,
            log_plots=log_plots,
            num_bins_cal_err=num_bins_cal_err,
        )
        self.save_hyperparameters(ignore=["loss"])
