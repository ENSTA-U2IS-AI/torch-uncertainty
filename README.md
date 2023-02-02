# Torch Uncertainty

[![Code style: black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)

## Installation

The package can be installed from PyPI or from source.

### From source

#### Installing Poetry

Installation guidelines for poetry are available on <https://python-poetry.org/docs/>. They boil down to executing the following command:

`curl -sSL https://install.python-poetry.org | python3 -`

#### Installing the package

Clone the repository with:

`https://github.com/ENSTA-U2IS/torch-uncertainty.git`

Create a new conda environment and activate it with:

`conda create -n uncertainty && conda activate uncertainty`

Install the package using poetry

`poetry install` or, for development, `poetry install --with dev`

Depending on your system, you may encounter an error. If so, kill the process and add `PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` at the beginning of your command.

## Credits
