# fmt: off
from pathlib import Path
from typing import Tuple, Union


# fmt: on
def get_version(
    root: Union[str, Path], version: int, checkpoint: Union[int, None] = None
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


# def packing(model: nn.Module, num_estimators: int, alpha: int, gamma: int):
#     """
#     Pack a model into a Packed-Ensemble.

#     Args:
#         model (nn.Module): The model to be packed.
#         num_estimators (int): The number of estimators in the ensemble.
#         alpha (int): _description_
#         gamma (int): _description_

#     Returns:
#         nn.Module: The packed model.
#     """
#     for attr_str in dir(model):
#         target_attr = getattr(model, attr_str)
#         if isinstance(target_attr, nn.Conv2d):
#             in_channels = target_attr.in_channels
#             out_channels = target_attr.out_channels
#             kernel_size = target_attr.kernel_size
#             stride = target_attr.stride
#             padding = target_attr.padding
#             dilation = target_attr.dilation
#             groups = target_attr.groups
#             bias = target_attr.bias is not None
#             device = target_attr.weight.device
#             dtype = target_attr.weight.dtype
#             setattr(
#                 model,
#                 attr_str,
#                 PackedConv2d(
#                     in_channels=in_channels * alpha,
#                     out_channels=out_channels * alpha,
#                     kernel_size=kernel_size,
#                     num_estimators=num_estimators,
#                     stride=stride,
#                     padding=padding,
#                     dilation=dilation,
#                     groups=groups * gamma,
#                     bias=bias,
#                     device=device,
#                     dtype=dtype,
#                 ),
#             )
#         elif isinstance(target_attr, nn.Linear):
#             in_features = target_attr.in_features
#             out_features = target_attr.out_features
#             bias = target_attr.bias is not None
#             device = target_attr.weight.device
#             dtype = target_attr.weight.dtype
#             setattr(
#                 model,
#                 attr_str,
#                 PackedLinear(
#                     in_features=in_features * alpha,
#                     out_features=out_features * alpha,
#                     num_estimators=num_estimators,
#                     bias=bias,
#                     device=device,
#                     dtype=dtype,
#                 ),
#             )

#         elif isinstance(target_attr, nn.BatchNorm2d):
#             num_features = target_attr.num_features
#             eps = target_attr.eps
#             momentum = target_attr.momentum
#             affine = target_attr.affine
#             track_running_stats = target_attr.track_running_stats
#             device = target_attr.weight.device
#             dtype = target_attr.weight.dtype
#             setattr(
#                 model,
#                 attr_str,
#                 nn.BatchNorm2d(
#                     num_features=num_features * alpha,
#                     eps=eps,
#                     momentum=momentum,
#                     affine=affine,
#                     track_running_stats=track_running_stats,
#                     device=device,
#                     dtype=dtype,
#                 ),
#             )
