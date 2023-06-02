# fmt: off
from argparse import ArgumentParser


# fmt: on
def add_resnet_specific_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--arch",
        type=int,
        choices=[18, 34, 50, 101, 152],
        default=18,
        help=f"Architecture of ResNet. Choose among: {[18, 34, 50, 101, 152]}",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=1,
        help="Number of groups",
    )
    return parser


def add_wideresnet_specific_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--groups",
        type=int,
        default=1,
        help="Number of groups",
    )
    return parser


def add_packed_specific_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Alpha for Packed-Ensembles",
    )
    parser.add_argument(
        "--gamma",
        type=int,
        default=1,
        help="Gamma for Packed-Ensembles",
    )
    return parser


def add_masked_specific_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Scale for Masksembles",
    )
    return parser
