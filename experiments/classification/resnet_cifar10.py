# fmt: off
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from torchinfo import summary

import numpy as np
from torch_uncertainty.baselines import ResNet
from torch_uncertainty.datamodules import CIFAR10DataModule
from torch_uncertainty.optimization_procedures import get_procedure
from torch_uncertainty.utils import get_version

# fmt: on
if __name__ == "__main__":
    root = Path(__file__).parent.absolute().parents[1]

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
    parser = CIFAR10DataModule.add_argparse_args(parser)
    parser = ResNet.add_model_specific_args(parser)
    args = parser.parse_args()

    # print(args)

    if isinstance(root, str):
        root = Path(root)

    if isinstance(args.seed, int):
        pl.seed_everything(args.seed)

    net_name = f"{args.version}-resnet{args.arch}-cifar10"

    # datamodule
    args.root = str(root / "data")
    dm = CIFAR10DataModule(**vars(args))

    # model
    model = ResNet(
        num_classes=dm.num_classes,
        in_channels=dm.num_channels,
        loss=nn.CrossEntropyLoss,
        optimization_procedure=get_procedure(f"resnet{args.arch}", "cifar10"),
        imagenet_structure=False,
        **vars(args),
    )

    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

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
