from pathlib import Path

from .base import ImageNetVariation


class ImageNetC(ImageNetVariation):
    """The corrupted ImageNet-C dataset."""

    url = [
        "https://zenodo.org/record/2235448/files/blur.tar",
        "https://zenodo.org/record/2235448/files/digital.tar",
        "https://zenodo.org/record/2235448/files/extra.tar",
        "https://zenodo.org/record/2235448/files/noise.tar",
        "https://zenodo.org/record/2235448/files/weather.tar",
    ]
    filename = ["blur.tar", "digital.tar", "extra.tar", "noise.tar", "weather.tar"]
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
        severity = kwargs.pop("shift_severity", 1)
        try:
            severity = int(severity)
        except Exception as e:
            raise ValueError(f"shift_severity must be an int in [1..5], got {severity!r}") from e
        if severity not in (1, 2, 3, 4, 5):
            raise ValueError(f"shift_severity must be in [1..5], got {severity}")

        super().__init__(**kwargs)

        sev_str = str(severity)
        filtered = [(p, t) for (p, t) in self.samples if Path(p).parts[-3] == sev_str]
        if not filtered:
            raise RuntimeError(
                f"ImageNet-C: no samples matched shift_severity={severity}. "
                "Check extraction under <root>/imagenet-c/<corruption>/<severity>/..."
            )

        self.samples = filtered
        self.imgs = filtered
        self.targets = [t for _, t in filtered]
        self.shift_severity = severity
