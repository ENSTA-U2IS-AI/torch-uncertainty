# fmt: off
from pathlib import Path
from typing import Tuple, Union


# fmt: on
def get_version(
    root: Union[str, Path], version: int, checkpoint: int = None
) -> Tuple[Path, Path]:
    """
    Find a compute the path to the checkpoint corresponding to the input
        parameters


    Args:
        root (Union[str, Path]): The root of the dataset containing the
            checkpoints.
        version (int): The version of the checkpoint
        checkpoint (int, optional): The number of the checkpoint. Defaults
            to None.

    Raises:
        Exception: if the checkpoint cannot be found.

    Returns:
        Tuple[Path, Path]: The path to the checkpoints and to its parameters.
    """
    if isinstance(root, str):
        root = Path(root)

    if (root / f"version_{version}").is_dir():
        version_folder = root / f"version_{version}"
        ckpt_folder = version_folder / "checkpoints"
        if checkpoint is None:
            ckpts = list(ckpt_folder.glob("*.ckpt"))
        else:
            ckpts = list(ckpt_folder.glob(f"epoch={checkpoint}-*.ckpt"))
    else:
        raise Exception(
            f"The directory {root}/version_{version} does not exist."
        )

    file = ckpts[0]
    return (file.resolve(), (version_folder / "hparams.yaml").resolve())
