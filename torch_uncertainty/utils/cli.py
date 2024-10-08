from collections.abc import Callable
from pathlib import Path
from typing import Any

from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning.pytorch import LightningDataModule, LightningModule, Trainer
from lightning.pytorch.cli import (
    ArgsType,
    LightningArgumentParser,
    LightningCLI,
    SaveConfigCallback,
)
from typing_extensions import override

from torch_uncertainty.utils.trainer import TUTrainer


class TUSaveConfigCallback(SaveConfigCallback):
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
                if file_exists:  # coverage: ignore
                    raise RuntimeError(
                        f"{self.__class__.__name__} expected {config_path} to NOT exist. Aborting to avoid overwriting"
                        " results of a previous run. You can delete the previous config file,"
                        " set `LightningCLI(save_config_callback=None)` to disable config saving,"
                        ' or set `LightningCLI(save_config_kwargs={"overwrite": True})` to overwrite the config file.'
                    )

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


class TULightningCLI(LightningCLI):
    def __init__(
        self,
        model_class: (
            type[LightningModule] | Callable[..., LightningModule] | None
        ) = None,
        datamodule_class: (
            type[LightningDataModule]
            | Callable[..., LightningDataModule]
            | None
        ) = None,
        save_config_callback: type[SaveConfigCallback]
        | None = TUSaveConfigCallback,
        save_config_kwargs: dict[str, Any] | None = None,
        trainer_class: type[Trainer] | Callable[..., Trainer] = TUTrainer,
        trainer_defaults: dict[str, Any] | None = None,
        seed_everything_default: bool | int = True,
        parser_kwargs: dict[str, Any] | dict[str, dict[str, Any]] | None = None,
        subclass_mode_model: bool = False,
        subclass_mode_data: bool = False,
        args: ArgsType = None,
        run: bool = True,
        auto_configure_optimizers: bool = True,
        eval_after_fit_default: bool = False,
    ) -> None:
        """Custom LightningCLI for torch-uncertainty.

        Args:
            model_class (type[LightningModule] | Callable[..., LightningModule] | None, optional): _description_. Defaults to None.
            datamodule_class (type[LightningDataModule] | Callable[..., LightningDataModule] | None, optional): _description_. Defaults to None.
            save_config_callback (type[SaveConfigCallback] | None, optional): _description_. Defaults to TUSaveConfigCallback.
            save_config_kwargs (dict[str, Any] | None, optional): _description_. Defaults to None.
            trainer_class (type[Trainer] | Callable[..., Trainer], optional): _description_. Defaults to Trainer.
            trainer_defaults (dict[str, Any] | None, optional): _description_. Defaults to None.
            seed_everything_default (bool | int, optional): _description_. Defaults to True.
            parser_kwargs (dict[str, Any] | dict[str, dict[str, Any]] | None, optional): _description_. Defaults to None.
            subclass_mode_model (bool, optional): _description_. Defaults to False.
            subclass_mode_data (bool, optional): _description_. Defaults to False.
            args (ArgsType, optional): _description_. Defaults to None.
            run (bool, optional): _description_. Defaults to True.
            auto_configure_optimizers (bool, optional): _description_. Defaults to True.
            eval_after_fit_default (bool, optional): _description_. Defaults to False.
        """
        self.eval_after_fit_default = eval_after_fit_default
        super().__init__(
            model_class,
            datamodule_class,
            save_config_callback,
            save_config_kwargs,
            trainer_class,
            trainer_defaults,
            seed_everything_default,
            parser_kwargs,
            subclass_mode_model,
            subclass_mode_data,
            args,
            run,
            auto_configure_optimizers,
        )

    def add_default_arguments_to_parser(
        self, parser: LightningArgumentParser
    ) -> None:
        """Adds default arguments to the parser."""
        parser.add_argument(
            "--eval_after_fit",
            action="store_true",
            default=self.eval_after_fit_default,
        )
        super().add_default_arguments_to_parser(parser)

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        super().add_arguments_to_parser(parser)
        parser.link_arguments("data.eval_ood", "model.eval_ood")
        parser.link_arguments("data.eval_shift", "model.eval_shift")
