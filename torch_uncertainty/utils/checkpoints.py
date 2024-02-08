from pathlib import Path


def get_version(
    root: str | Path, version: int, checkpoint: int | None = None
) -> tuple[Path, Path]:
    """Find the path to the checkpoint corresponding to the version.

    Args:
        root (Union[str, Path]): The root of the dataset containing the
            checkpoints.
        version (int): The version of the checkpoint.
        checkpoint (int, optional): The number of the checkpoint. Defaults
            to None.

    Raises:
        FileNotFoundError: if the checkpoint cannot be found.

    Returns:
        Tuple[Path, Path]: The path to the checkpoints and to its parameters.
    """
    root = Path(root)

    if (root / f"version_{version}").is_dir():
        version_folder = root / f"version_{version}"
        ckpt_folder = version_folder / "checkpoints"
        if checkpoint is None:
            ckpts = list(ckpt_folder.glob("*.ckpt"))
        else:
            ckpts = list(ckpt_folder.glob(f"epoch={checkpoint}-*.ckpt"))
    else:
        raise FileNotFoundError(
            f"The directory {root}/version_{version} does not exist."
        )

    file = ckpts[0]
    return (file.resolve(), (version_folder / "hparams.yaml").resolve())
