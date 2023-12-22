"""Creates the checkpoint for hugging face."""

import argparse
from pathlib import Path

import torch

parser = argparse.ArgumentParser(
    prog="to_hub_format",
    description="Post-process the checkpoints before the upload to HuggingFace",
)
parser.add_argument(
    "--name", type=str, required=True, help="path to the checkpoint"
)
parser.add_argument(
    "--path", type=Path, required=True, help="path to the checkpoint"
)
parser.add_argument(
    "--version", type=int, default=0, help="path to the checkpoint"
)

args = parser.parse_args()

if not args.path.exists():
    raise ValueError("File does not exist")

model = torch.load(args.path)["state_dict"]
model = {key.replace("model.", ""): val.cpu() for key, val in model.items()}

output_name = args.name
if args.version != 0:
    output_name += "_" + str(args.version)
output_name += ".ckpt"

torch.save(model, output_name)
