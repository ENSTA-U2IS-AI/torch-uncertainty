# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from datetime import datetime

import tu_sphinx_theme

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "TorchUncertainty"

copyright = (  # noqa: A001
    f"{datetime.now().year!s}, Adrien Lafage and Olivier Laurent"
)
author = "Adrien Lafage and Olivier Laurent"
release = "0.3.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    # "sphinxcontrib.katex",
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    # "sphinx.ext.intersphinx",  # for links to the API in the tutorials
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_codeautolink",
    "sphinx_gallery.gen_gallery",
    # "sphinx_gallery.load_style",
]

sphinx_gallery_conf = {
    "examples_dirs": ["../../auto_tutorials_source"],
    "gallery_dirs": "auto_tutorials",
    "filename_pattern": r"tutorial_",
    "plot_gallery": "True",
    "promote_jupyter_magic": True,
    "backreferences_dir": None,
    "first_notebook_cell": (
        "# For tips on running notebooks in Google Colab, see\n"
        "# https://pytorch.org/tutorials/beginner/colab\n"
        "%matplotlib inline"
    ),
    "reference_url": {
        "sphinx_gallery": None,
    },
}

# Use both the docstrings of the init and the class
autoclass_content = "both"

autosummary_generate = True
napoleon_use_ivar = True

# Disable docstring inheritance
autodoc_inherit_docstrings = False

# Disable displaying type annotations, these can be very verbose
autodoc_typehints = "none"

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "tu_sphinx_theme"
html_theme_path = [tu_sphinx_theme.get_html_theme_path()]

html_theme_options = {
    "logo_url": "https://torch-uncertainty.github.io/",
    "menu": [
        {
            "name": "GitHub",
            "url": "https://github.com/ENSTA-U2IS-AI/torch-uncertainty",
        }
    ],
    "pytorch_project": "tutorials",
}

html_static_path = ["_static"]
html_style = "css/custom.css"
