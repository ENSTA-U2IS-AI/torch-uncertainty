#! /usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="torch-uncertainty",
    version="0.1.3",
    description="A PyTorch Library for benchmarking and leveraging efficient"
    "predictive uncertainty quantification techniques.",
    author="Adrien Lafage & Olivier Laurent",
    author_email="olivier.laurent@ensta-paris.fr",
    url="https://torch-uncertainty.github.io/",
    install_requires=[
        "python>=3.10",
        "pytorch-lightning=^1.9.0",
        "tensorboard=^2.6.0",
        "einops=^0.6.0",
        "torchinfo=^1.7.1",
        "torchvision>=0.14",
        "timm=^0.6.12",
        "scipy=^1.10.0",
        "huggingface_hub=^0.14.1",
        "pandas=^2.0.3",
    ],
    packages=find_packages(exclude=["experiments", "tests"]),
)
