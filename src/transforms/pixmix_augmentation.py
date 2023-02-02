# fmt: off
import numpy as np
from PIL import Image, ImageEnhance, ImageOps


# fmt: on

# TODO: change this !
IMAGE_SIZE = 32


def get_ab(beta):
    if np.random.random() < 0.5:
        a = np.float32(np.random.beta(beta, 1))
        b = np.float32(np.random.beta(1, beta))
    else:
        a = 1 + np.float32(np.random.beta(1, beta))
        b = -np.float32(np.random.beta(1, beta))
    return a, b


def add(img1, img2, beta):
    a, b = get_ab(beta)
    img1, img2 = img1 * 2 - 1, img2 * 2 - 1
    out = a * img1 + b * img2
    return (out + 1) / 2


def multiply(img1, img2, beta):
    a, b = get_ab(beta)
    img1, img2 = img1 * 2, img2 * 2
    out = (img1**a) * (img2.clip(1e-37) ** b)
    return out / 2


mixings = [add, multiply]


def int_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval .
    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.
    Returns:
      An int that results from scaling `maxval` according to `level`.
    """
    return int(level * maxval / 10)


def float_parameter(level, maxval):
    """Helper function to scale `val` between 0 and maxval.
    Args:
      level: Level of the operation that will be between [0, `PARAMETER_MAX`].
      maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.
    Returns:
      A float that results from scaling `maxval` according to `level`.
    """
    return float(level) * maxval / 10.0


def sample_level(n):
    return np.random.uniform(low=0.1, high=n)


def autocontrast(pil_img, _):
    return ImageOps.autocontrast(pil_img)


def equalize(pil_img, _):
    return ImageOps.equalize(pil_img)


def posterize(pil_img, level):
    level = int_parameter(sample_level(level), 4)
    return ImageOps.posterize(pil_img, 4 - level)


def rotate(pil_img, level):
    degrees = int_parameter(sample_level(level), 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return pil_img.rotate(degrees, resample=Image.BILINEAR)


def solarize(pil_img, level):
    level = int_parameter(sample_level(level), 256)
    return ImageOps.solarize(pil_img, 256 - level)


def shear_x(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, level, 0, 0, 1, 0),
        resample=Image.BILINEAR,
    )


def shear_y(pil_img, level):
    level = float_parameter(sample_level(level), 0.3)
    if np.random.uniform() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, 0, 0, level, 1, 0),
        resample=Image.BILINEAR,
    )


def translate_x(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, 0, level, 0, 1, 0),
        resample=Image.BILINEAR,
    )


def translate_y(pil_img, level):
    level = int_parameter(sample_level(level), IMAGE_SIZE / 3)
    if np.random.random() > 0.5:
        level = -level
    return pil_img.transform(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.AFFINE,
        (1, 0, 0, 0, 1, level),
        resample=Image.BILINEAR,
    )


# operation that overlaps with ImageNet-C's test set
def color(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Color(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def contrast(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Contrast(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def brightness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Brightness(pil_img).enhance(level)


# operation that overlaps with ImageNet-C's test set
def sharpness(pil_img, level):
    level = float_parameter(sample_level(level), 1.8) + 0.1
    return ImageEnhance.Sharpness(pil_img).enhance(level)


augmentations = [
    autocontrast,
    equalize,
    posterize,
    rotate,
    solarize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
]

augmentations_all = [
    autocontrast,
    equalize,
    posterize,
    rotate,
    solarize,
    shear_x,
    shear_y,
    translate_x,
    translate_y,
    color,
    contrast,
    brightness,
    sharpness,
]


def augment_input(image, aug_severity: int = 1, all_ops: bool = False):
    aug_list = augmentations_all if all_ops else augmentations
    op = np.random.choice(aug_list)
    return op(image.copy(), aug_severity)


def pixmix(mixing_set, mixing_iterations: int = 4, mixing_severity: int = 3):
    def _pixmix(orig):
        if np.random.random() < 0.5:
            mixed = augment_input(orig)
        else:
            mixed = orig

        for _ in range(np.random.randint(mixing_iterations + 1)):
            if np.random.random() < 0.5:
                aug_image_copy = augment_input(orig)
            else:
                aug_image_copy = np.random.choice(len(mixing_set))

            mixed_op = np.random.choice([add, multiply])
            mixed = mixed_op(
                np.array(mixed), np.array(aug_image_copy), mixing_severity
            )
            mixed = np.clip(mixed, 0, 1)
        return mixed

    return _pixmix
