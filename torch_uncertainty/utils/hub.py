from pathlib import Path

import torch
import yaml
from huggingface_hub import hf_hub_download
from huggingface_hub.utils._errors import EntryNotFoundError
from safetensors.torch import load_file


def load_hf(
    weight_id: str, version: int = 0
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Load a model from the HuggingFace hub.

    Args:
        weight_id (str): The id of the model to load.
        version (int): The id of the version when there are several on HF.

    Returns:
        tuple[dict[str, torch.Tensor], dict[str, str]]: The model weights and config.

    Note - License:
        TorchUncertainty's weights are released under the Apache 2.0 license.
    """
    repo_id = f"torch-uncertainty/{weight_id}"

    # Load the weights
    pickle = True
    if version is None or version == 0:
        filename = f"{weight_id}.ckpt"
    else:
        filename = f"{weight_id}_{version}.ckpt"
    try:
        weight_path = hf_hub_download(repo_id=repo_id, filename=filename)
    except EntryNotFoundError as not_pt:
        pickle = False
        filename = f"{weight_id}_{version}.safetensors"
        try:
            weight_path = hf_hub_download(repo_id=repo_id, filename=filename)
        except EntryNotFoundError:
            raise ValueError(
                f"Model {weight_id}_{version} not found on HuggingFace."
            ) from not_pt

    if pickle:
        weight = torch.load(weight_path, map_location=torch.device("cpu"))
    else:
        weight = load_file(weight_path, device="cpu")

    if "state_dict" in weight:  # coverage: ignore
        weight = weight["state_dict"]

    # Load the config
    config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
    config = yaml.safe_load(Path(config_path).read_text())
    return weight, config
