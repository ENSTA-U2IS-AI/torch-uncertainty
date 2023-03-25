# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import pytorch_sphinx_theme

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Torch Uncertainty"
copyright = "2023, Adrien Lafage and Olivier Laurent"
author = "Adrien Lafage and Olivier Laurent"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinxcontrib.katex",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
]

sphinx_gallery_conf = {
    "examples_dirs": ["../../tutorials"],
    "gallery_dirs": "auto_tutorials",
    "filename_pattern": r"pe_",
}


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

html_theme = 'pytorch_sphinx_theme'
html_theme_path = [pytorch_sphinx_theme.get_html_theme_path()]

html_theme_options = {
    "logo_url": "https://torch-uncertainty.github.io/",
    "menu": [
        {
            "name": "GitHub",
            "url": "https://github.com/ENSTA-U2IS/torch-uncertainty",
        }
    ],
}

html_static_path = ["_static"]
html_css_files = [
    'https://cdn.jsdelivr.net/npm/katex@0.10.0-beta/dist/katex.min.css',
    'css/custom.css'
]
