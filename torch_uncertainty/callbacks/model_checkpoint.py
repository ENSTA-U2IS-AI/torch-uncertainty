from lightning.pytorch.callbacks import Checkpoint, ModelCheckpoint


# FIXME: this is incomplete
class TUClsCheckpoint(Checkpoint):
    """Custom ModelCheckpoint class for saving the best model based on validation loss."""

    def __init__(self):
        super().__init__()
        self.callbacks = {
            "acc": ModelCheckpoint(
                filename="{epoch}-{step}-val_acc={val/cls/Acc:.2f}",
                monitor="val/cls/Acc",
                mode="max",
            ),
            "ece": ModelCheckpoint(
                filename="{epoch}-{step}-val_ece={val/cal/ECE:.2f}",
                monitor="val/cal/ECE",
                mode="min",
            ),
            "brier": ModelCheckpoint(
                filename="{epoch}-{step}-val_brier={val/cls/Brier:.2f}",
                monitor="val/cls/Brier",
                mode="min",
            ),
        }
