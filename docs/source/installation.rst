Installation
============

.. role:: bash(code)
    :language: bash


You can install the package from PyPI or from source. Choose the latter if you
want to access the files included the `experiments <https://github.com/ENSTA-U2IS/torch-uncertainty/tree/main/experiments>`_
folder or if you want to contribute to the project.


From PyPI
---------

Install the package via pip: 

.. parsed-literal::

    pip install torch-uncertainty

To update the package, run:

.. parsed-literal::

    pip install -U torch-uncertainty 

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
    cd torch-uncertainty

Create a new conda environment and activate it:

.. parsed-literal::

    conda create -n uncertainty python=3.10
    conda activate uncertainty

Install the package using poetry:

.. parsed-literal::

    poetry install
    # or, for development
    poetry install --with dev


Depending on your system, you may encounter poetry errors. If so, kill the 
process and add :bash:`PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring`
at the beginning of every :bash:`poetry` command.

To update the package, run:

.. parsed-literal::

    git pull && poetry update
