# ruff: noqa: F401
from argparse import ArgumentParser, Namespace
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torchinfo import summary

import numpy as np

from .datamodules.abstract import AbstractDataModule
from .utils import get_version


def init_args(
    network: Any = None,
    datamodule: Optional[Type[pl.LightningDataModule]] = None,
) -> Namespace:
    parser = ArgumentParser("torch-uncertainty")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to make the training deterministic.",
    )
    parser.add_argument(
        "--test",
        type=int,
        default=None,
        help="Run in test mode. Set to the checkpoint version number to test.",
    )
    parser.add_argument(
        "--summary",
        dest="summary",
        action="store_true",
        help="Print model summary",
    )
    parser.add_argument("--log_graph", dest="log_graph", action="store_true")
    parser.add_argument(
        "--channels_last",
        action="store_true",
        help="Use channels last memory format",
    )
    parser.add_argument(
        "--enable_resume",
        action="store_true",
        help="Allow resuming the training (save optimizer's states)",
    )
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="logs/",
        help="Directory to store experiment files",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="",
        help="Name of the experiment folder",
    )
    parser.add_argument(
        "--opt_temp_scaling", action="store_true", default=False
    )
    parser.add_argument(
        "--val_temp_scaling", action="store_true", default=False
    )
    parser = pl.Trainer.add_argparse_args(parser)
    if network is not None:
        parser = network.add_model_specific_args(parser)

    if datamodule is not None:
        parser = datamodule.add_argparse_args(parser)

    return parser.parse_args()


def cli_main(
    network: pl.LightningModule | list[pl.LightningModule],
    datamodule: AbstractDataModule | list[AbstractDataModule],
    root: Union[Path, str],
    net_name: str,
    args: Namespace,
) -> list[Dict]:
    if isinstance(root, str):
        root = Path(root)

    if isinstance(datamodule, list):
        training_task = datamodule[0].dm.training_task
    else:
        training_task = datamodule.training_task
    if training_task == "classification":
        monitor = "hp/val_acc"
        mode = "max"
    elif training_task == "regression":
        monitor = "hp/val_mse"
        mode = "min"
    else:
        raise ValueError("Unknown problem type.")

    if args.test is None and args.max_epochs is None:
        print(
            "Setting max_epochs to 1 for testing purposes. Set max_epochs "
            "manually to train the model."
        )
        args.max_epochs = 1

    if isinstance(args.seed, int):
        pl.seed_everything(args.seed, workers=True)

    if args.channels_last:
        if isinstance(network, list):
            for i in range(len(network)):
                network[i] = network[i].to(memory_format=torch.channels_last)
        else:
            network = network.to(memory_format=torch.channels_last)

    if args.use_cv:
        test_values = []
        for i in range(len(datamodule)):
            print(
                f"Starting fold {i} out of {args.train_over} of a {args.n_splits}-fold CV."
            )

            # logger
            tb_logger = TensorBoardLogger(
                str(root),
                name=net_name,
                default_hp_metric=False,
                log_graph=args.log_graph,
                version=f"fold_{i}",
            )

            # callbacks
            save_checkpoints = ModelCheckpoint(
                dirpath=tb_logger.log_dir,
                monitor=monitor,
                mode=mode,
                save_last=True,
                save_weights_only=not args.enable_resume,
            )

    if args.summary:
        summary(network, input_size=list(datamodule.input_shape).insert(0, 1))
        test_values = {}
    elif args.test is not None:  # coverage: ignore
        if args.test >= 0:
            ckpt_file, _ = get_version(
                root=(root / "logs" / net_name), version=args.test
            )

            # Select the best model, monitor the lr and stop if NaN
            callbacks = [
                save_checkpoints,
                LearningRateMonitor(logging_interval="step"),
                EarlyStopping(
                    monitor=monitor, patience=np.inf, check_finite=True
                ),
            ]

            trainer = pl.Trainer.from_argparse_args(
                args,
                callbacks=callbacks,
                logger=tb_logger,
                deterministic=(args.seed is not None),
                inference_mode=not (
                    args.opt_temp_scaling or args.val_temp_scaling
                ),
            )
            trainer.fit(network[i], datamodule[i])
            test_values.append(
                trainer.test(datamodule=datamodule[i], ckpt_path="last")[0]
            )

        all_test_values = defaultdict(list)
        for test_value in test_values:
            for key in test_value:
                all_test_values[key].append(test_value[key])

        avg_test_values = {}
        for key in all_test_values:
            avg_test_values[key] = np.mean(all_test_values[key])

        return [avg_test_values]
    else:
        # logger
        tb_logger = TensorBoardLogger(
            str(root / "logs"),
            name=net_name,
            default_hp_metric=False,
            log_graph=args.log_graph,
            version=args.test,
        )

        # callbacks
        save_checkpoints = ModelCheckpoint(
            monitor=monitor,
            mode=mode,
            save_last=True,
            save_weights_only=not args.enable_resume,
        )

        # Select the best model, monitor the lr and stop if NaN
        callbacks = [
            save_checkpoints,
            LearningRateMonitor(logging_interval="step"),
            EarlyStopping(monitor=monitor, patience=np.inf, check_finite=True),
        ]

        # trainer
        trainer = pl.Trainer.from_argparse_args(
            args,
            callbacks=callbacks,
            logger=tb_logger,
            deterministic=(args.seed is not None),
            inference_mode=not (args.opt_temp_scaling or args.val_temp_scaling),
        )
        if args.summary:
            summary(
                network,
                input_size=list(datamodule.input_shape).insert(0, 1),
            )
            test_values = [{}]
        elif args.test is not None:
            if args.test >= 0:
                ckpt_file, _ = get_version(
                    root=(root / net_name), version=args.test
                )
                test_values = trainer.test(
                    network, datamodule=datamodule, ckpt_path=str(ckpt_file)
                )
            else:
                test_values = trainer.test(network, datamodule=datamodule)
        else:
            # training and testing
            trainer.fit(network, datamodule)
            if args.fast_dev_run is False:
                test_values = trainer.test(
                    datamodule=datamodule, ckpt_path="best"
                )
            else:
                test_values = [{}]
        return test_values
