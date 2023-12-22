from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download


def load_hf(weight_id: str, version: int = 0) -> tuple[torch.Tensor, dict]:
    """Load a model from the HuggingFace hub.

    Args:
        weight_id (str): The id of the model to load.
        version (int): The id of the version when there are several on HF.

    Returns:
        Tuple[Tensor, Dict]: The model weights and config.

    Note - License:
        TorchUncertainty's weights are released under the Apache 2.0 license.
    """
    repo_id = f"torch-uncertainty/{weight_id}"

    # Load the weights
    if version is None or version == 0:
        filename = f"{weight_id}.ckpt"
    else:
        filename = f"{weight_id}_{version}.ckpt"

    weight_path = hf_hub_download(repo_id=repo_id, filename=filename)
    weight = torch.load(weight_path, map_location=torch.device("cpu"))
    if "state_dict" in weight:  # coverage: ignore
        weight = weight["state_dict"]

    # Load the config
    config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
    config = yaml.safe_load(Path(config_path).read_text())

    return weight, config
