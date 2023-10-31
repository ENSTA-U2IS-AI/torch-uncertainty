from .base import ImageNetVariation


class ImageNetR(ImageNetVariation):
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar"
    filename = "imagenet-r.tar"
    tgz_md5 = "a61312130a589d0ca1a8fca1f2bd3337"
    dataset_name = "imagenet-r"
