# Torch Uncertainty

[![tests](https://github.com/ENSTA-U2IS/torch-uncertainty/actions/workflows/run-tests.yml/badge.svg?branch=main&event=push)](https://github.com/ENSTA-U2IS/torch-uncertainty/actions/workflows/run-tests.yml) [![Code Coverage](https://img.shields.io/codecov/c/github/ENSTA-U2IS/torch-uncertainty.svg)](https://codecov.io/gh/ENSTA-U2IS/torch-uncertainty) [![Code style: black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)

_Torch Uncertainty_ is a package designed to help you leverage uncertainty quantification techniques and make your neural networks more reliable. It is based on PyTorch Lightning to handle multi-GPU training and inference and automatic logging through tensorboard.

---

This package provides a multi-level API, including:
- ready-to-train baselines on research datasets, such as CIFAR and ImageNet
- baselines available for training on your datasets
- layers available for use in your networks

## Installation

The package can be installed from PyPI or from source.

### From PyPI (available soon)

Install the package via pip: `pip install torch-uncertainty`

### From source

#### Installing Poetry

Installation guidelines for poetry are available on <https://python-poetry.org/docs/>. They boil down to executing the following command:

`curl -sSL https://install.python-poetry.org | python3 -`

#### Installing the package

Clone the repository:

`git clone https://github.com/ENSTA-U2IS/torch-uncertainty.git`

Create a new conda environment and activate it with:

`conda create -n uncertainty && conda activate uncertainty`

Install the package using poetry:

`poetry install torch-uncertainty` or, for development, `poetry install torch-uncertainty --with dev`

Depending on your system, you may encounter errors. If so, kill the process and add `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` at the beginning of every `poetry install` commands.

#### Contributing

In case that you would like to contribute, install from source and add the pre-commit hooks with `pre-commit install`

## Getting Started and Documentation

Please find the documentation at [torch-uncertainty.github.io](https://torch-uncertainty.github.io).

A quickstart is available at [torch-uncertainty.github.io/quickstart](https://torch-uncertainty.github.io/quickstart.html).

## Implemented baselines

To date, the following baselines are implemented:

- Deep Ensembles
- Masksembles
- Packed-Ensembles


## Awesome Torch repositories

You may find a lot of information about modern uncertainty estimation techniques on the [Awesome Uncertainty in Deep Learning](https://github.com/ENSTA-U2IS/awesome-uncertainty-deeplearning).

## References

