Contributing
============

.. role:: bash(code)
    :language: bash


TorchUncertainty is in early development stage. We are looking for
contributors to help us build a comprehensive library for uncertainty
quantification in PyTorch.

We are particularly open to any comment that you would have on this project.
Specifically, we are open to changing these guidelines as the project evolves.

The scope of TorchUncertainty
-----------------------------

TorchUncertainty can host any method - if possible linked to a paper - and
roughly contained in the following fields:
* uncertainty quantification in general, including Bayesian deep learning,
Monte Carlo dropout, ensemble methods, etc.
* Out-of-distribution detection methods
* Applications (e.g. object detection, segmentation, etc.)

Common guidelines
-----------------

Clean development install of TorchUncertainty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you are interested in contributing to torch_uncertainty, we first advise you
to follow the following steps to reproduce a clean development environment
ensuring continuous integration does not break.

1. Install poetry on your workstation.
2. Clone the repository.
3. Install torch-uncertainty in editable mode poetry with dev packages:
   :bash:`poetry install --with dev`
4. Install pre-commit hooks with :bash:`pre-commit install`

If you have issues with poetry, add :bash:`PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`
at the beginning of every :bash:`poetry` command.

Build the documentation locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build the documentation, reinstall TorchUncertainty with the packages of the docs
group:

.. parsed-literal::

    poetry install --with dev,docs

Then navigate to :bash:`./docs` and build the documentation with:

.. parsed-literal::
    
    make html


Optionally, specify :bash:`html-noplot` instead of :bash:`html` to avoid running the tutorials.

Guidelines
^^^^^^^^^^

We are using ruff for code formatting, linting, and imports (as a drop-in
replacement for black, isort, and flake8). The pre-commit hooks will ensure
that your code is properly formatted and linted before committing.

To make your changes, create a branch on a personal fork, whose name is not
dev nor main.

Before submitting a final pull request, that we will review, please try your
best not to reduce the code coverage and do document your code.

Try to include an emoji at the start of each commit message following the suggestions
from `this page <https://gist.github.com/parmentf/035de27d6ed1dce0b36a>`_.

If you implement a method, please add a reference to the corresponding paper in the 
`references page <https://torch-uncertainty.github.io/references.html>`_. Also,
remember to add TorchUncertainty to the list of libraries implementing this reference
on `PapersWithCode <https://paperswithcode.com>`_.

Datasets & Datamodules
^^^^^^^^^^^^^^^^^^^^^^

We intend to include datamodules for the most popular datasets only.

Post-processing methods
^^^^^^^^^^^^^^^^^^^^^^^

For now, we intend to follow scikit-learn style API for post-processing
methods (except that we use a validation dataset instead of a numpy array).
You may get inspiration from the already implemented
`temperature-scaling <https://github.com/ENSTA-U2IS/torch-uncertainty/blob/dev/torch_uncertainty/post_processing/calibration/temperature_scaler.py>`_.


License
-------

If you feel that the current license is an obstacle to your contribution, let
us know, and we may reconsider. However, the models’ weights hosted on Hugging
Face are likely to stay Apache 2.0.
