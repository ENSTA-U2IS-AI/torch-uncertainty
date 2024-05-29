from torchvision.transforms import InterpolationMode


def interpolation_modes_from_str(val: str) -> InterpolationMode:
    val = val.lower()
    inverse_modes_mapping = {
        "nearest": InterpolationMode.NEAREST,
        "bilinear": InterpolationMode.BILINEAR,
        "bicubic": InterpolationMode.BICUBIC,
        "box": InterpolationMode.BOX,
        "hamming": InterpolationMode.HAMMING,
        "lanczos": InterpolationMode.LANCZOS,
    }
    return inverse_modes_mapping[val]
