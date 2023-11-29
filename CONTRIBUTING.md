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

### Clean development install of TorchUncertainty

If you are interested in contributing to torch_uncertainty, we first advise you
to follow the following steps to reproduce a clean development environment
ensuring that continuous integration does not break.

1. Check that you have PyTorch already installed on your system
2. Clone the repository
3. Install torch-uncertainty in editable mode with the dev packages:

```sh
python3 -m pip install -e .[dev]
```

4. Install pre-commit hooks with:

```sh
pre-commit install
```

### Build the documentation locally

To build the documentation, reinstall TorchUncertainty with the packages of the docs
group:

```sh
python3 -m pip install -e .[dev,docs]
```

Then navigate to `./docs` and build the documentation with:

```sh
make html
```

Optionally, specify `html-noplot` instead of `html` to avoid running the tutorials.

### Guidelines

#### Commits

We are use `ruff` for code formatting, linting, and imports (as a drop-in
replacement for `black`, `isort`, and `flake8`). The `pre-commit` hooks will
ensure that your code is properly formatted and linted before committing.

Please ensure that the tests are passing on your machine before pushing on a
PR. This will avoid multiplying the number featureless commits. To do this,
run, at the root of the folder:

```sh
python3 -m pytest tests
```

Try to include an emoji at the start of each commit message following the suggestions
from [this page](https://gist.github.com/parmentf/035de27d6ed1dce0b36a).

#### Pull requests

To make your changes, create a branch on a personal fork and create a PR when your contribution
is mostly finished or if you need help.

Check that your PR complies with the following conditions:

- The name of your branch is not `main` nor `dev` (see issue #58)
- Your PR does not reduce the code coverage
- Your code is documented: the function signatures are typed, and the main functions have clear
docstrings
- Your code is mostly original, and the parts coming from licensed sources are explicitly
stated as such
- If you implement a method, please add a reference to the corresponding paper in the
["References" page](https://torch-uncertainty.github.io/references.html).
- Also, remember to add TorchUncertainty to the list of libraries implementing this reference
on [PapersWithCode](https://paperswithcode.com).

If you need help to implement a method, increase the coverage, or solve ruff-raised errors,
create the PR with the `need-help` flag and explain your problems in the comments. A maintainer
will do their best to help you.

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
