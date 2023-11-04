Installation
============

.. role:: bash(code)
    :language: bash


You can install the package from PyPI or from source. Choose the latter if you
want to access the files included the `experiments <https://github.com/ENSTA-U2IS/torch-uncertainty/tree/main/experiments>`_
folder or if you want to contribute to the project.


From PyPI
---------

Check that you have PyTorch (cpu or gpu) installed on your system. Then, install
the package via pip: 

.. parsed-literal::

    pip install torch-uncertainty

To update the package, run:

.. parsed-literal::

    pip install -U torch-uncertainty 

From source
-----------

To install the project from source, you can use pip directly.

Again, with PyTorch already installed, clone the repository with:

.. parsed-literal::

    git clone https://github.com/ENSTA-U2IS/torch-uncertainty.git
    cd torch-uncertainty

Create a new conda environment and activate it:

.. parsed-literal::

    conda create -n uncertainty python=3.10
    conda activate uncertainty

Install the package using pip in editable mode:

.. parsed-literal::

    pip install -e .

If PyTorch is not installed, the latest version will be installed automatically.
