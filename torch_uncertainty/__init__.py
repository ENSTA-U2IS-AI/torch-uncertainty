# fmt: off
# flake8: noqa
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Optional, Type, Union

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
    network: Optional[Type[pl.LightningModule]] = None,
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

    parser = pl.Trainer.add_argparse_args(parser)
    if network is not None:
        parser = network.add_model_specific_args(parser)

    if datamodule is not None:
        parser = datamodule.add_argparse_args(parser)

    return parser.parse_args()


def cli_main(
    network: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    root: Union[Path, str],
    net_name: str,
    args: Namespace,
) -> Dict:
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
        pl.seed_everything(args.seed, workers=True)

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
    )

    if args.summary:
        summary(network, input_size=list(datamodule.input_shape).insert(0, 1))
        test_values = {}
    elif args.test is not None:
        if args.test >= 0:
            ckpt_file, _ = get_version(
                root=(root / "logs" / net_name), version=args.test
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
            test_values = trainer.test(datamodule=datamodule, ckpt_path="best")
        else:
            test_values = {}
    return test_values
