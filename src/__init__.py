# fmt: off
from argparse import ArgumentParser
from pathlib import Path
from typing import Type, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torchinfo import summary

import numpy as np

from .routines.classification import ClassificationSingle
from .utils import get_version


# fmt: on
def cli_main(
    network: Type[ClassificationSingle],
    datamodule: Type[pl.LightningDataModule],
    root: Union[Path, str],
    net_name: str,
) -> None:
    parser = ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test", type=int, default=None)
    parser.add_argument("--summary", dest="summary", action="store_true")
    parser.add_argument("--log_graph", dest="log_graph", action="store_true")
    parser.add_argument("--profile", dest="profile", action="store_true")
    parser.add_argument("--multi-gpu", action="store_true")

    parser = pl.Trainer.add_argparse_args(parser)
    parser = datamodule.add_argparse_args(parser)
    parser = network.add_model_specific_args(parser)
    args = parser.parse_args()

    if args.deterministic and args.seed is None:
        print("Setting seed to 0.")
        args.__setattr__("seed", 0)

    if args.seed:
        pl.seed_everything(args.seed, workers=True)

    if args.profile:
        print(
            "Profiling will leak memory and increase the computational time."
            "Do not launch long lasting trainings with the profile flag."
        )

    if isinstance(root, str):
        root = Path(root)

    # datamodule
    args.root = str(root / args.root)
    dm = datamodule(**vars(args))

    # model
    model = network(args)

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
        monitor="hp/val_acc",
        mode="max",
        save_last=True,
        save_weights_only=True,
    )

    # Select the best model, monitor the lr and stop if NaN
    callbacks = [
        save_checkpoints,
        LearningRateMonitor(logging_interval="step"),
        EarlyStopping(monitor="hp/val_nll", patience=np.inf, check_finite=True),
    ]

    # trainer
    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        logger=tb_logger,
        deterministic=(args.seed is not None),
    )

    if args.summary:
        summary(model, input_size=model.example_input_array.shape)
    elif args.test is not None:
        ckpt_file, _ = get_version(
            root=(root / "logs" / net_name), version=args.test
        )
        trainer.test(model, datamodule=dm, ckpt_path=str(ckpt_file))
    else:
        # training and testing
        trainer.fit(model, dm)
        trainer.test(datamodule=dm, ckpt_path="best")
