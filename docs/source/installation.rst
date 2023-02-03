Installation
============

.. role:: bash(code)
    :language: bash

The package can be installed from PyPI or from source.

From PyPI
---------

Install the package via pip: 

.. parsed-literal::

    pip install torch-uncertainty

From source
-----------

**Installing Poetry**

Installation guidelines for poetry are available `here <https://python-poetry.org/docs/>`_.
They boil down to executing the following command:

.. parsed-literal::
    
    curl -sSL https://install.python-poetry.org | python3 -

**Installing the package**

Clone the repository with:

.. parsed-literal::

    git clone https://github.com/ENSTA-U2IS/torch-uncertainty.git

Create a new conda environment and activate it with:

.. parsed-literal::

    conda create -n uncertainty
    conda activate uncertainty

Install the package using poetry:

.. parsed-literal::

    poetry install
    # or, for development
    poetry install --with dev


.. note::
    Depending on your system, you may encounter an error. If so, kill the
    process and add :bash:`PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`
    at the beginning of your command.