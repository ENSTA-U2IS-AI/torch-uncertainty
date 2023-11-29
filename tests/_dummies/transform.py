class DummyTransform:
    """Dummy transform for testing purposes."""

    def __init__(self) -> None:
        pass

    def __call__(self, img):
        return img
