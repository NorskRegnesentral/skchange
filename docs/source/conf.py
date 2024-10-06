"""Configuration file for the Sphinx documentation builder."""

import datetime
import sys

import skchange

# -- Project information

project = "skchange"
current_year = datetime.datetime.now().year
copyright = f"2023 - {current_year} (BSD-3-Clause License)"
author = "Tveten"

version = f"v{skchange.__version__}"

# -- General configuration

extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
intersphinx_mapping = {
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "sktime": ("https://scikit-learn.org/stable/", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
}
intersphinx_disabled_domains = ["std"]

templates_path = ["_templates"]

# -- Options for HTML output

html_theme = "pydata_sphinx_theme"
