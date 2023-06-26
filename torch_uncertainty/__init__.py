# fmt: off
# flake8: noqa
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Type, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torchinfo import summary

import numpy as np

from .utils import get_version


# fmt: on
def init_args(
    network: Type[pl.LightningModule], datamodule: Type[pl.LightningDataModule]
) -> Namespace:
    parser = ArgumentParser("torch-uncertainty")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test", type=int, default=None)
    parser.add_argument("--summary", dest="summary", action="store_true")
    parser.add_argument("--log_graph", dest="log_graph", action="store_true")
    parser.add_argument(
        "--channels_last",
        action="store_true",
        help="Use channels last memory format",
    )

    parser = pl.Trainer.add_argparse_args(parser)
    parser = datamodule.add_argparse_args(parser)
    parser = network.add_model_specific_args(parser)
    args = parser.parse_args()

    return args


def cls_main(
    network: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    root: Union[Path, str],
    net_name: str,
    args: Namespace,
) -> None:
    if isinstance(root, str):
        root = Path(root)

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
        pl.seed_everything(args.seed)

    if args.channels_last:
        network = network.to(memory_format=torch.channels_last)

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
        save_weights_only=True,
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
    )

    if args.summary:
        summary(network, input_size=list(datamodule.input_shape).insert(0, 1))
    elif args.test is not None:
        ckpt_file, _ = get_version(
            root=(root / "logs" / net_name), version=args.test
        )
        trainer.test(network, datamodule=datamodule, ckpt_path=str(ckpt_file))
    else:
        # training and testing
        trainer.fit(network, datamodule)
        trainer.test(datamodule=datamodule, ckpt_path="best")
