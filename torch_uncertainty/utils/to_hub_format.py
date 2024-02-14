"""Creates the checkpoint for Hugging Face."""

import argparse
from pathlib import Path

import torch
from safetensors.torch import save_file

torch.set_grad_enabled(False)

parser = argparse.ArgumentParser(
    prog="to_hub_format",
    description="Post-process the checkpoints before the upload to HuggingFace",
)
parser.add_argument(
    "--name", type=str, required=True, help="name of the checkpoint"
)
parser.add_argument(
    "--path", type=Path, required=True, help="path to the checkpoint"
)
parser.add_argument(
    "--version", type=int, default=0, help="version of the checkpoint"
)
parser.add_argument(
    "--safe", action="store_true", help="whether to use safetensors"
)
parser.add_argument(
    "--fp16", action="store_true", help="whether to use fp16 for the checkpoint"
)

args = parser.parse_args()

if not args.path.exists():
    raise ValueError("File does not exist")

dtype = torch.float16 if args.fp16 else torch.float32
model = torch.load(args.path)["state_dict"]
model = {
    key.replace("model.", ""): val.to(device="cpu", dtype=dtype)
    for key, val in model.items()
}

output_name = args.name
if args.version != 0:
    output_name += "_" + str(args.version)
output_name += ".ckpt" if not args.safe else ".safetensors"

if args.safe:
    save_file(model, output_name)
else:
    torch.save(model, output_name)
