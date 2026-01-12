from datetime import timedelta
from pathlib import Path
from typing import Literal

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from torch import Tensor


class CompoundCheckpoint(ModelCheckpoint):
    def __init__(
        self,
        compound_metric_dict: dict,
        dirpath: str | Path | None = None,
        verbose: bool = False,
        save_last: bool | None | Literal["link"] = None,
        save_top_k: int = 1,
        save_weights_only: bool = False,
        mode: str = "min",
        every_n_train_steps: int | None = None,
        train_time_interval: timedelta | None = None,
        every_n_epochs: int | None = None,
        save_on_train_epoch_end: bool | None = None,
        enable_version_counter: bool = True,
    ) -> None:
        """Save the checkpoints maximizing or minimizing a given linear form on the metric values."""
        self.compound_metric_dict = compound_metric_dict
        super().__init__(
            dirpath=dirpath,
            filename="epoch={epoch}-step={step}-compound={compound_metric:.3f}",
            monitor="compound_metric",
            verbose=verbose,
            save_last=save_last,
            save_top_k=save_top_k,
            save_weights_only=save_weights_only,
            mode=mode,
            auto_insert_metric_name=False,
            every_n_train_steps=every_n_train_steps,
            train_time_interval=train_time_interval,
            every_n_epochs=every_n_epochs,
            save_on_train_epoch_end=save_on_train_epoch_end,
            enable_version_counter=enable_version_counter,
        )

    def _monitor_candidates(self, trainer: Trainer) -> dict[str, Tensor]:
        monitor_candidates = super()._monitor_candidates(trainer)
        result = torch.tensor(
            0.0, dtype=torch.float32, device=next(iter(monitor_candidates.values())).device
        )
        for metric, factor in self.compound_metric_dict.items():
            result += factor * monitor_candidates[metric].to(
                dtype=result.dtype, device=result.device
            )
        monitor_candidates["compound_metric"] = result
        return monitor_candidates
