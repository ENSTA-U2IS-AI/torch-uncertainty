# fmt: off
from pathlib import Path
from typing import Dict, Tuple

import torch
import yaml
from huggingface_hub import hf_hub_download


# fmt: on
def load_hf(weight_id: str) -> Tuple[torch.Tensor, Dict]:
    """Load a model from the HuggingFace hub.

    Args:
        weight_id (str): The id of the model to load.

    Returns:
        Tuple[Tensor, Dict]: The model weights and config.

    Note - License:
        TorchUncertainty's weights are released under the Apache 2.0 license.
    """
    repo_id = f"torch-uncertainty/{weight_id}"

    # Load the weights
    weight_path = hf_hub_download(repo_id=repo_id, filename=f"{weight_id}.ckpt")
    weight = torch.load(weight_path, map_location=torch.device("cpu"))
    if "state_dict" in weight:  # coverage: ignore
        weight = weight["state_dict"]

    # Load the config
    config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
    config = yaml.safe_load(Path(config_path).read_text())

    return weight, config
