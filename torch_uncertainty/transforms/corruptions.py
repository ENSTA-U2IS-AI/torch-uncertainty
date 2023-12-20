"""Adapted from https://github.com/hendrycks/robustness."""

import cv2
import numpy as np
import torch
from skimage.filters import gaussian
from skimage.util import random_noise
from torch import nn


class GaussianNoise(nn.Module):
    def __init__(self, severity: int = 1):
        super().__init__()
        if not (0 <= severity <= 5):
            raise ValueError("Severity must be between 0 and 5.")
        self.severity = severity
        self.scale = [0, 0.04, 0.06, 0.08, 0.09, 0.10][severity]

    def forward(self, img):
        if self.severity == 0:
            return img
        return torch.clip(torch.normal(img, self.scale), 0, 1)

    def __repr__(self):
        """Printable representation."""
        return self.__class__.__name__ + f"(severity={self.severity})"


class ShotNoise(nn.Module):
    def __init__(self, severity: int = 1):
        super().__init__()
        if not (0 <= severity <= 5):
            raise ValueError("Severity must be between 0 and 5.")
        self.severity = severity
        self.scale = [500, 250, 100, 75, 50][severity - 1]

    def forward(self, img):
        if self.severity == 0:
            return img
        return torch.clip(torch.poisson(img * self.scale) / self.scale, 0, 1)

    def __repr__(self):
        """Printable representation."""
        return self.__class__.__name__ + f"(severity={self.severity})"


class ImpulseNoise(nn.Module):
    def __init__(self, severity: int = 1):
        super().__init__()
        if not (0 <= severity <= 5):
            raise ValueError("Severity must be between 0 and 5.")
        self.severity = severity
        self.scale = [0, 0.01, 0.02, 0.03, 0.05, 0.07][severity]

    def forward(self, img):
        if self.severity == 0:
            return img
        return torch.clip(
            torch.as_tensor(random_noise(img, mode="s&p", amount=self.scale)),
            0,
            1,
        )

    def __repr__(self):
        """Printable representation."""
        return self.__class__.__name__ + f"(severity={self.severity})"


class SpeckleNoise(nn.Module):
    def __init__(self, severity: int = 1):
        super().__init__()
        if not (0 <= severity <= 5):
            raise ValueError("Severity must be between 0 and 5.")
        self.severity = severity
        self.scale = [0.06, 0.1, 0.12, 0.16, 0.2][severity - 1]

    def forward(self, img):
        if self.severity == 0:
            return img
        return torch.clip(
            img + img * torch.normal(img.shape, scale=self.scale),
            0,
            1,
        )

    def __repr__(self):
        """Printable representation."""
        return self.__class__.__name__ + f"(severity={self.severity})"


class GaussianBlur(nn.Module):
    def __init__(self, severity: int = 1):
        super().__init__()
        if not (0 <= severity <= 5):
            raise ValueError("Severity must be between 0 and 5.")
        self.severity = severity
        self.sigma = [0.4, 0.6, 0.7, 0.8, 1.0][severity - 1]

    def forward(self, img):
        if self.severity == 0:
            return img
        return torch.clip(
            torch.as_tensor(gaussian(img, sigma=self.sigma)),
            0,
            1,
        )

    def __repr__(self):
        """Printable representation."""
        return self.__class__.__name__ + f"(severity={self.severity})"


class GlassBlur(nn.Module):  # TODO: batch
    def __init__(self, severity: int = 1):
        super().__init__()
        if not (0 <= severity <= 5):
            raise ValueError("Severity must be between 0 and 5.")
        self.severity = severity
        self.sigma = [0.05, 0.25, 0.4, 0.25, 0.4][severity - 1]
        self.max_delta = 1
        self.iterations = [1, 1, 1, 2, 2][severity - 1]

    def forward(self, img):
        img_size = img.shape
        img = torch.as_tensor(gaussian(img, sigma=self.sigma))
        for _ in range(self.iterations):
            for h in range(img_size[0] - self.max_delta, self.max_delta, -1):
                for w in range(
                    img_size[1] - self.max_delta, self.max_delta, -1
                ):
                    dx, dy = torch.randint(
                        -self.max_delta, self.max_delta, size=(2,)
                    )
                    h_prime, w_prime = h + dy, w + dx
                    img[h, w], img[h_prime, w_prime] = (
                        img[h_prime, w_prime],
                        img[h, w],
                    )
        return torch.clip(
            torch.as_tensor(gaussian(img, sigma=self.sigma)), 0, 1
        )

    def __repr__(self):
        """Printable representation."""
        return self.__class__.__name__ + f"(severity={self.severity})"


def disk(radius, alias_blur=0.1, dtype=np.float32):
    if radius <= 8:
        size = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        size = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    xs, ys = np.meshgrid(size, size)
    aliased_disk = np.array((xs**2 + ys**2) <= radius**2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


class DefocusBlur(nn.Module):
    def __init__(self, severity: int = 1):
        super().__init__()
        if not (0 <= severity <= 5):
            raise ValueError("Severity must be between 0 and 5.")
        self.severity = severity
        self.radius = [0.3, 0.4, 0.5, 1, 1.5][severity - 1]
        self.alias_blur = [0.4, 0.5, 0.6, 0.2, 0.1][severity - 1]

    def forward(self, img):
        if self.severity == 0:
            return img
        img = np.array(img)
        channels = [
            torch.as_tensor(
                cv2.filter2D(
                    img[d, :, :],
                    -1,
                    disk(self.radius, alias_blur=self.alias_blur),
                )
            )
            for d in range(3)
        ]
        return torch.clip(torch.stack(channels), 0, 1)

    def __repr__(self):
        """Printable representation."""
        return self.__class__.__name__ + f"(severity={self.severity})"
