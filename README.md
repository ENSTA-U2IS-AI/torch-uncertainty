<div align="center">

![Torch Uncertainty Logo](https://github.com/ENSTA-U2IS/torch-uncertainty/blob/main/docs/source/_static/images/torch_uncertainty.png)

[![pypi](https://img.shields.io/pypi/v/torch_uncertainty.svg)](https://pypi.python.org/pypi/torch_uncertainty) [![tests](https://github.com/ENSTA-U2IS/torch-uncertainty/actions/workflows/run-tests.yml/badge.svg?branch=main&event=push)](https://github.com/ENSTA-U2IS/torch-uncertainty/actions/workflows/run-tests.yml) [![Code Coverage](https://img.shields.io/codecov/c/github/ENSTA-U2IS/torch-uncertainty.svg)](https://codecov.io/gh/ENSTA-U2IS/torch-uncertainty) [![Code style: black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)
</div>

_Torch Uncertainty_ is a package designed to help you leverage uncertainty quantification techniques and make your neural networks more reliable. It is based on PyTorch Lightning to handle multi-GPU training and inference and automatic logging through tensorboard.

---

This package provides a multi-level API, including:
- ready-to-train baselines on research datasets, such as CIFAR and ImageNet
- baselines available for training on your datasets
- layers available for use in your networks

## Reference

This package also contains the official implementation of Packed-Ensembles.

If you find the corresponding models interesting, please consider citing our [paper](https://arxiv.org/abs/2210.09184):
	
    @inproceedings{laurent2023packed,
        title={Packed-Ensembles for Efficient Uncertainty Estimation},
        author={Laurent, Olivier and Lafage, Adrien and Tartaglione, Enzo and Daniel, Geoffrey and Martinez, Jean-Marc and Bursuc, Andrei and Franchi, Gianni},
        booktitle={ICLR},
        year={2023}
    }


## Installation

The package can be installed from PyPI or from source.

### From PyPI

Install the package via pip: 
```sh
pip install torch-uncertainty
```

### From source with Poetry

#### Installing Poetry

Installation guidelines for poetry are available on <https://python-poetry.org/docs/>. They boil down to executing the following command:
```sh
curl -sSL https://install.python-poetry.org | python3 -
```

#### Installing the package

Clone the repository:

```sh
git clone https://github.com/ENSTA-U2IS/torch-uncertainty.git
```

Create a new conda environment and activate it with:

```sh
conda create -n uncertainty python=3.10 && conda activate uncertainty
```

Install the package using poetry:

```sh
poetry install torch-uncertainty
```
or, for development,

```sh
poetry install torch-uncertainty --with dev
```

Depending on your system, you may encounter poetry-related errors. If so, kill the process and add `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` at the beginning of every `poetry install` commands.

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

## Other References


