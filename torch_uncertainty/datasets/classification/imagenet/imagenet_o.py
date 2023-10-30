from .base import ImageNetVariation


class ImageNetO(ImageNetVariation):
    url = "https://people.eecs.berkeley.edu/~hendrycks/imagenet-o.tar"
    filename = "imagenet-o.tar"
    tgz_md5 = "86bd7a50c1c4074fb18fc5f219d6d50b"
    dataset_name = "imagenet-o"
