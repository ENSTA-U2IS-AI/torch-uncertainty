from .base import ImageNetVariation


class ImageNetC(ImageNetVariation):
    """The corrupted ImageNet-C dataset.

    References:
        Benchmarking neural network robustness to common corruptions and
        perturbations. Dan Hendrycks and Thomas Dietterich.
        In ICLR, 2019.
    """

    url = [
        "https://zenodo.org/record/2235448/files/blur.tar",
        "https://zenodo.org/record/2235448/files/digital.tar",
        "https://zenodo.org/record/2235448/files/extra.tar",
        "https://zenodo.org/record/2235448/files/noise.tar",
        "https://zenodo.org/record/2235448/files/weather.tar",
    ]
    filename = [
        "blur.tar",
        "digital.tar",
        "extra.tar",
        "noise.tar",
        "weather.tar",
    ]
    tgz_md5 = [
        "2d8e81fdd8e07fef67b9334fa635e45c",
        "89157860d7b10d5797849337ca2e5c03",
        "d492dfba5fc162d8ec2c3cd8ee672984",
        "e80562d7f6c3f8834afb1ecf27252745",
        "33ffea4db4d93fe4a428c40a6ce0c25d",
    ]
    dataset_name = "imagenet-c"
    root_appendix = "imagenet-c"

    def __init__(self, **kwargs) -> None:
        """Initializes the ImageNetC dataset class.

        This is a subclass of ImageNetVariation that supports additional keyword arguments.

        Args:
            kwargs: Additional keyword arguments passed to the superclass, including:

                - root (str): Root directory of the datasets.
                - split (str, optional): For API consistency. Defaults to ``None``.
                - transform (callable, optional): A function/transform that takes in a PIL image and
                  returns a transformed version. E.g., transforms.RandomCrop. Defaults to ``None``.
                - target_transform (callable, optional): A function/transform that takes in the target
                  and transforms it. Defaults to ``None``.
                - download (bool, optional): If ``True``, downloads the dataset from the internet
                  and puts it in the root directory. If the dataset is already downloaded, it is
                  not downloaded again. Defaults to ``False``.
        """
        super().__init__(**kwargs)
