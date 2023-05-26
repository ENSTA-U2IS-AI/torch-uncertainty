# fmt: off
from argparse import ArgumentParser
from pathlib import Path

import pytorch_lightning as pl
import torch.nn as nn

from torch_uncertainty import main
from torch_uncertainty.baselines.batched import BatchedResNet
from torch_uncertainty.baselines.masked import MaskedResNet
from torch_uncertainty.baselines.packed import PackedResNet
from torch_uncertainty.baselines.standard import ResNet
from torch_uncertainty.datamodules import CIFAR10DataModule, CIFAR100DataModule
from torch_uncertainty.optimization_procedures import get_procedure

# fmt: on

if __name__ == "__main__":
    root = Path(__file__).parent.absolute().parents[1]

    parser = ArgumentParser("torch-uncertainty")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--test", type=int, default=None)
    parser.add_argument("--summary", dest="summary", action="store_true")
    parser.add_argument("--log_graph", dest="log_graph", action="store_true")
    parser.add_argument(
        "--type", choices=["standard", "packed", "masked", "batched"]
    )
    parser.add_argument(
        "--model", choices=["resnet18", "resnet50", "wideresnet28x10"]
    )
    parser.add_argument("--data", choices=["cifar10", "cifar100", "imagenet"])

    parser = pl.Trainer.add_argparse_args(parser)
    parser = ResNet.add_model_specific_args(parser)
    parser = CIFAR10DataModule.add_argparse_args(parser)
    parser = CIFAR100DataModule.add_argparse_args(parser)
    args = parser.parse_args()

    if args.data == "cifar10":
        datamodule = CIFAR10DataModule
    elif args.data == "cifar100":
        datamodule = CIFAR100DataModule
    elif args.data == "imagenet":
        raise NotImplementedError("ImageNet not yet implemented")
    else:
        raise ValueError(f"Unknown dataset: {args.data}")

    if args.type == "standard":
        model_type = ResNet
    elif args.type == "masked":
        model_type = MaskedResNet
    elif args.type == "batched":
        model_type = BatchedResNet
    elif args.type == "packed":
        model_type = PackedResNet
    else:
        raise ValueError(f"Unknown model type: {args.type}")

    main(
        model_type,
        datamodule,
        nn.CrossEntropyLoss,
        get_procedure(args.model, args.data),
        root,
        f"{args.model}_{args.data}",
    )
