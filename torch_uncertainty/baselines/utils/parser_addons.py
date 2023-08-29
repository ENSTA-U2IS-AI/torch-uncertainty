# fmt: off
from argparse import ArgumentParser


# fmt: on
def add_resnet_specific_args(parser: ArgumentParser) -> ArgumentParser:
    # style_choices = ["cifar", "imagenet", "robust"]
    archs = [18, 34, 50, 101, 152]
    parser.add_argument(
        "--arch",
        type=int,
        choices=archs,
        default=18,
        help=f"Architecture of ResNet. Choose among: {archs}",
    )
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.0,
        help="Dropout rate",
    )
    # parser.add_argument(
    #     "--style",
    #     type=str,
    #     choices=style_choices,
    #     default="imagenet",
    #     help=f"Variation of ResNet. Choose among: {style_choices}",
    # )
    parser.add_argument(
        "--groups",
        type=int,
        default=1,
        help="Number of groups",
    )
    return parser


def add_vgg_specific_args(parser: ArgumentParser) -> ArgumentParser:
    # style_choices = ["cifar", "imagenet", "robust"]
    archs = [11, 13, 16, 19]
    parser.add_argument(
        "--arch",
        type=int,
        choices=archs,
        default=11,
        help=f"Architecture of VGG. Choose among: {archs}",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=1,
        help="Number of groups",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    return parser


def add_wideresnet_specific_args(parser: ArgumentParser) -> ArgumentParser:
    # style_choices = ["cifar", "imagenet"]
    # parser.add_argument(
    #     "--style",
    #     type=str,
    #     choices=style_choices,
    #     default="imagenet",
    #     help=f"Variation of ResNet. Choose among: {style_choices}",
    # )
    parser.add_argument(
        "--groups",
        type=int,
        default=1,
        help="Number of groups",
    )
    return parser


def add_mlp_specific_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    return parser


def add_packed_specific_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--alpha",
        type=int,
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


def add_mimo_specific_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--rho",
        type=float,
        default=0.0,
        help="Rho for MIMO",
    )
    parser.add_argument(
        "--batch_repeat",
        type=int,
        default=1,
        help="Batch repeat for MIMO",
    )
    return parser
