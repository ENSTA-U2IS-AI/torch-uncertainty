#! /usr/bin/env python
# flake8: noqa
from setuptools import find_packages, setup

setup(
    name="torch_uncertainty",
    version="0.1.5",
    description="TorchUncertainty: A maintained and collaborative PyTorch"
    "Library for benchmarking and leveraging predictive uncertainty"
    "quantification techniques.",
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
        "ruff=^0.1.0",
        "pytest-cov=^4.0.0",
        "pre-commit=^3.0.4",
        "pre-commit-hooks=^4.4.0",
        "cli-test-helpers=^3.2.0",
        "sphinx=^5.1.3",
        "pytorch-sphinx-theme=git+https://github.com/torch-uncertainty/pytorch_sphinx_theme",
        "sphinx-copybutton=^0.5.1",
        "sphinx-gallery=^0.12.2",
        "matplotlib=^3.7.1",
        "sphinx-design=^0.3.0",
    ],
    packages=find_packages(exclude=["experiments", "tests"]),
)
