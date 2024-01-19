from pathlib import Path

from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.cli import SaveConfigCallback
from typing_extensions import override


class MySaveConfigCallback(SaveConfigCallback):
    @override
    def setup(
        self, trainer: Trainer, pl_module: LightningModule, stage: str
    ) -> None:
        if self.already_saved:
            return

        if self.save_to_log_dir and stage == "fit":
            log_dir = trainer.log_dir  # this broadcasts the directory
            assert log_dir is not None
            config_path = Path(log_dir) / self.config_filename
            fs = get_filesystem(log_dir)

            if not self.overwrite:
                # check if the file exists on rank 0
                file_exists = (
                    fs.isfile(config_path) if trainer.is_global_zero else False
                )
                # broadcast whether to fail to all ranks
                file_exists = trainer.strategy.broadcast(file_exists)
                if file_exists:
                    # TODO: complete error description
                    raise RuntimeError("TODO")

            if trainer.is_global_zero:
                fs.makedirs(log_dir, exist_ok=True)
                self.parser.save(
                    self.config,
                    config_path,
                    skip_none=False,
                    overwrite=self.overwrite,
                    multifile=self.multifile,
                )
        if trainer.is_global_zero:
            self.save_config(trainer, pl_module, stage)
            self.already_saved = True

        self.already_saved = trainer.strategy.broadcast(self.already_saved)
