from .base import ImageNetVariation


class ImageNetA(ImageNetVariation):
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"
    filename = "imagenet-a.tar"
    tgz_md5 = "c3e55429088dc681f30d81f4726b6595"
    dataset_name = "imagenet-a"

    def __init__(self, **kwargs) -> None:
        """Initializes the ImageNetA dataset class.

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
