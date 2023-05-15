from huggingface_hub import hf_hub_download


def load_hf(weight_id: str):
    weights = hf_hub_download(
        repo_id=f"torch-uncertainty/{weight_id}", filename=f"{weight_id}.ckpt"
    )
    return weights
