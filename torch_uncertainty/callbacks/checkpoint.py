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
    def __init__(self):
        super().__init__()
        self.callbacks = {
            "acc": ModelCheckpoint(
                filename="{epoch}-{step}-{val_cls_Acc:.2f}",
                monitor="val_cls_Acc",
                mode="max",
            ),
            "ece": ModelCheckpoint(
                filename="{epoch}-{step}-{val_cal_ECE:.2f}",
                monitor="val_cal_ECE",
                mode="min",
            ),
            "brier": ModelCheckpoint(
                filename="{epoch}-{step}-{val_cls_Brier:.2f}",
                monitor="val_cls_Brier",
                mode="min",
            ),
            "nll": ModelCheckpoint(
                filename="{epoch}-{step}-{val_cls_NLL:.2f}",
                monitor="val_cls_NLL",
                mode="min",
            ),
        }

    @property
    def best_model_path(self) -> str:
        return self.callbacks["acc"].best_model_path


class TUSegCheckpoint(TUCheckpoint):
    def __init__(self):
        super().__init__()
        self.callbacks = {
            "miou": ModelCheckpoint(
                filename="{epoch}-{step}-{val_seg_mIoU:.2f}",
                monitor="val_seg_mIoU",
                mode="max",
            ),
            "ece": ModelCheckpoint(
                filename="{epoch}-{step}-{val_cal_ECE:.2f}",
                monitor="val_cal_ECE",
                mode="min",
            ),
            "brier": ModelCheckpoint(
                filename="{epoch}-{step}-{val_seg_Brier:.2f}",
                monitor="val_seg_Brier",
                mode="min",
            ),
            "nll": ModelCheckpoint(
                filename="{epoch}-{step}-{val_seg_NLL:.2f}",
                monitor="val_seg_NLL",
                mode="min",
            ),
        }

    @property
    def best_model_path(self) -> str:
        return self.callbacks["miou"].best_model_path


class TURegCheckpoint(TUCheckpoint):
    def __init__(self, probabilistic: bool = False):
        super().__init__()
        self.callbacks = {
            "mse": ModelCheckpoint(
                filename="{epoch}-{step}-{val_reg_MSE:.2f}",
                monitor="val_reg_MSE",
                mode="min",
            ),
        }

        if probabilistic:
            self.callbacks["nll"] = ModelCheckpoint(
                filename="{epoch}-{step}-{val_reg_NLL:.2f}",
                monitor="val_reg_NLL",
                mode="min",
            )

    @property
    def best_model_path(self) -> str:
        return self.callbacks["mse"].best_model_path
