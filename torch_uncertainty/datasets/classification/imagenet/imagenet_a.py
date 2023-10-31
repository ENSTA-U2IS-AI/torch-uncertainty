from .base import ImageNetVariation


class ImageNetA(ImageNetVariation):
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar"
    filename = "imagenet-a.tar"
    tgz_md5 = "c3e55429088dc681f30d81f4726b6595"
    dataset_name = "imagenet-a"
