# Contributing to TorchUncertainty

TorchUncertainty is in early development stage. We are looking for
contributors to help us build a comprehensive library for uncertainty
quantification in PyTorch.

We are particularly open to any comment that you would have on this project.
Specifically, we are open to changing these guidelines as the project evolves.

## The scope of TorchUncertainty

TorchUncertainty can host every method - if possible linked to a paper -
roughly contained in the following fields:

- uncertainty quantification in general, including Bayesian deep learning,
Monte Carlo dropout, ensemble methods, etc.
- Out-of-distribution detection methods
- Applications (e.g. object detection, segmentation, etc.)

## Common guidelines

If you are interested in contributing to torch_uncertainty, we first advise you
to follow the following steps to reproduce a clean development environment
ensuring that continuous integration does not break.

1. Install poetry on your workstation.
2. Clone the repository.
3. Install torch-uncertainty in editable mode poetry with dev packages:

```sh
poetry install --with dev
```

4. Install pre-commit hooks with:

```sh
pre-commit install
```

We are using `black` for code formatting, `flake8` for linting, and `isort` for the
imports. The `pre-commit` hooks will ensure that your code is properly formatted
and linted before committing.

Before submitting a final pull request, that we will review, please try your
best not to reduce the code coverage and document your code.

If you implement a method, please add a reference to the corresponding paper in the ["References" page](https://torch-uncertainty.github.io/references.html).

### Datasets & Datamodules

We intend to include datamodules for the most popular datasets only.

### Post-processing methods

For now, we intend to follow scikit-learn style API for post-processing
methods (except that we use a validation dataset for now). You can get
inspiration from the already implemented
[temperature-scaling](https://github.com/ENSTA-U2IS/torch-uncertainty/blob/dev/torch_uncertainty/post_processing/calibration/temperature_scaler.py).

## License

If you feel that the current license is an obstacle to your contribution, let us
know, and we may reconsider. However, the models’ weights are likely to stay
Apache 2.0.
