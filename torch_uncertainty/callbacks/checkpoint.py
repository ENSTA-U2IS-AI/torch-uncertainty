from typing import Any

from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.callbacks import Checkpoint, ModelCheckpoint
from typing_extensions import override


class TUCheckpoint(Checkpoint):
    callbacks: dict[str, Checkpoint]

    @override
    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: str) -> None:
        for callback in self.callbacks.values():
            callback.setup(trainer=trainer, pl_module=pl_module, stage=stage)

    @override
    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for callback in self.callbacks.values():
            callback.on_train_start(trainer=trainer, pl_module=pl_module)

    @override
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: dict,
        batch: dict,
        batch_idx: int,
    ) -> None:
        for callback in self.callbacks.values():
            callback.on_train_batch_end(
                trainer=trainer,
                pl_module=pl_module,
                outputs=outputs,
                batch=batch,
                batch_idx=batch_idx,
            )

    @override
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for callback in self.callbacks.values():
            callback.on_train_epoch_end(trainer=trainer, pl_module=pl_module)

    @override
    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        for callback in self.callbacks.values():
            callback.on_validation_epoch_end(trainer=trainer, pl_module=pl_module)

    @override
    def state_dict(self) -> dict[str, dict[str, Any]]:
        return {key: callback.state_dict() for key, callback in self.callbacks.items()}

    @override
    def load_state_dict(self, state_dict: dict[str, dict[str, Any]]) -> None:
        for key, callback in self.callbacks.items():
            callback.load_state_dict(state_dict=state_dict[key])

    @property
    def best_model_path(self) -> str: ...


class TUClsCheckpoint(TUCheckpoint):
    def __init__(self) -> None:
        """Keep multiple checkpoints corresponding to the best classification metric values."""
        super().__init__()
        self.callbacks = {
            "acc": ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_acc={val/cls/Acc:.3f}",
                monitor="val/cls/Acc",
                mode="max",
                auto_insert_metric_name=False,
            ),
            "ece": ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_ece={val/cal/ECE:.3f}",
                monitor="val/cal/ECE",
                mode="min",
                auto_insert_metric_name=False,
            ),
            "brier": ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_brier={val/cls/Brier:.3f}",
                monitor="val/cls/Brier",
                mode="min",
                auto_insert_metric_name=False,
            ),
            "nll": ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_nll={val/cls/NLL:.3f}",
                monitor="val/cls/NLL",
                mode="min",
                auto_insert_metric_name=False,
            ),
        }

    @property
    def best_model_path(self) -> str:
        return self.callbacks["acc"].best_model_path


class TUSegCheckpoint(TUCheckpoint):
    def __init__(self) -> None:
        """Keep multiple checkpoints corresponding to the best segmentation metric values."""
        super().__init__()
        self.callbacks = {
            "miou": ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_miou={val/seg/mIoU:.3f}",
                monitor="val/seg/mIoU",
                mode="max",
                auto_insert_metric_name=False,
            ),
            "ece": ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_ece={val/cal/ECE:.3f}",
                monitor="val/cal/ECE",
                mode="min",
                auto_insert_metric_name=False,
            ),
            "brier": ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_brier={val/seg/Brier:.3f}",
                monitor="val/seg/Brier",
                mode="min",
                auto_insert_metric_name=False,
            ),
            "nll": ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_nll={val/seg/NLL:.3f}",
                monitor="val/seg/NLL",
                mode="min",
                auto_insert_metric_name=False,
            ),
        }

    @property
    def best_model_path(self) -> str:
        return self.callbacks["miou"].best_model_path


class TURegCheckpoint(TUCheckpoint):
    def __init__(self, probabilistic: bool = False) -> None:
        """Keep multiple checkpoints corresponding to the best regression metric values."""
        super().__init__()
        self.callbacks = {
            "mse": ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_mse={val/reg/MSE:.3f}",
                monitor="val/reg/MSE",
                mode="min",
                auto_insert_metric_name=False,
            ),
        }

        if probabilistic:
            self.callbacks["nll"] = ModelCheckpoint(
                filename="epoch={epoch}-step={step}-val_nll={val/reg/NLL:.3f}",
                monitor="val/reg/NLL",
                mode="min",
                auto_insert_metric_name=False,
            )

    @property
    def best_model_path(self) -> str:
        return self.callbacks["mse"].best_model_path
