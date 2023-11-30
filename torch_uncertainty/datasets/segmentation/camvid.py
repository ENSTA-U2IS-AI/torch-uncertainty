from collections.abc import Callable
from typing import NamedTuple

from torchvision.datasets import VisionDataset


class CamVidClass(NamedTuple):
    name: str
    index: int
    color: tuple[int, int, int]


class CamVid(VisionDataset):
    # Notes: some classes are not used here
    classes = [
        CamVidClass("sky", 0, (128, 128, 128)),
        CamVidClass("building", 1, (128, 0, 0)),
        CamVidClass("pole", 2, (192, 192, 128)),
        CamVidClass("road_marking", 3, (255, 69, 0)),
        CamVidClass("road", 4, (128, 64, 128)),
        CamVidClass("pavement", 5, (60, 40, 222)),
        CamVidClass("tree", 6, (128, 128, 0)),
        CamVidClass("sign_symbol", 7, (192, 128, 128)),
        CamVidClass("fence", 8, (64, 64, 128)),
        CamVidClass("car", 9, (64, 0, 128)),
        CamVidClass("pedestrian", 10, (64, 64, 0)),
        CamVidClass("bicyclist", 11, (0, 128, 192)),
        CamVidClass("unlabelled", 12, (0, 0, 0)),
    ]

    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        transforms: Callable | None = None,
    ) -> None:
        """`CamVid <http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/>`_ Dataset."""
        super().__init__(root, transforms, transform, target_transform)
