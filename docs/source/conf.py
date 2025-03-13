"""Configuration file for the Sphinx documentation builder."""

import datetime
import sys

import plotly.io as pio

import skchange

pio.renderers.default = "notebook"  # Needed for plotly to render in Sphinx

# -- Project information

project = "skchange"
current_year = datetime.datetime.now().year
copyright = f"2023 - {current_year} (BSD-3-Clause License)"
author = "Tveten"

CURRENT_VERSION = f"v{skchange.__version__}"

intersphinx_mapping = {
    "python": (f"https://docs.python.org/{sys.version_info.major}", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "sktime": ("https://scikit-learn.org/stable/", None),
    "numba": ("https://numba.readthedocs.io/en/stable/", None),
}
intersphinx_disabled_domains = ["std"]


# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.autosectionlabel",
    "numpydoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.linkcode",  # link to GitHub source code via linkcode_resolve()
    "sphinx_gallery.load_style",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinx_issues",
    "sphinx.ext.doctest",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "nbsphinx",
]

# Notebook options
nbsphinx_allow_errors = True
nbsphinx_execute = "always"
nbsphinx_codecell_lexer = "ipython3"

html_static_path = ["_static"]
html_css_files = ["custom_notebook.css"]

# Recommended by sphinx_design when using the MyST Parser
myst_enable_extensions = ["colon_fence"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# The main toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    ".ipynb_checkpoints",
    "Thumbs.db",
    ".DS_Store",
]

add_module_names = False

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# see http://stackoverflow.com/q/12206334/562769
numpydoc_show_class_members = True
# this is needed for some reason...
# see https://github.com/numpy/numpydoc/issues/69
numpydoc_class_members_toctree = False

numpydoc_validation_checks = {"all"}

# generate autosummary even if no references
autosummary_generate = True

# Members and inherited-members default to showing methods and attributes from a
# class or those inherited.
# Member-order orders the documentation in the order of how the members are defined in
# the source code.
autodoc_default_options = {
    "members": True,
    "inherited-members": True,
    "member-order": "bysource",
}

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# When building HTML using the sphinx.ext.mathjax (enabled by default),
# Myst-Parser injects the tex2jax_ignore (MathJax v2) and mathjax_ignore (MathJax v3)
# classes in to the top-level section of each MyST document, and adds some default
# configuration. This ensures that MathJax processes only math, identified by the
# dollarmath and amsmath extensions, or specified in math directives. We here silence
# the corresponding warning that this override happens.
suppress_warnings = ["myst.mathjax"]


def linkcode_resolve(domain, info):
    """Return URL to source code corresponding.

    Parameters
    ----------
    domain : str
    info : dict

    Returns
    -------
    url : str
    """

    def find_source():
        # try to find the file and line number, based on code from numpy:
        # https://github.com/numpy/numpy/blob/main/doc/source/conf.py#L286
        obj = sys.modules[info["module"]]
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
        import inspect
        import os

        fn = inspect.getsourcefile(obj)
        fn = os.path.relpath(fn, start=os.path.dirname(skchange.__file__))
        source, lineno = inspect.getsourcelines(obj)
        return fn, lineno, lineno + len(source) - 1

    if domain != "py" or not info["module"]:
        return None
    try:
        filename = "skchange/%s#L%d-L%d" % find_source()
    except Exception:
        filename = info["module"].replace(".", "/") + ".py"
    return f"https://github.com/NorskRegnesentral/skchange/tree/{CURRENT_VERSION}/{filename}"


# -- Options for HTML output

html_theme = "pydata_sphinx_theme"
html_title = "skchange"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NorskRegnesentral/skchange",
            "icon": "fab fa-github",
        },
    ],
    "show_prev_next": False,
    "use_edit_page_button": False,
    "navbar_start": ["navbar-logo"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}
html_context = {
    "github_user": "NorskRegnesentral",
    "github_repo": "skchange",
    "github_version": "main",
    "doc_path": "docs/source/",
}
html_sidebars = {
    "**": ["sidebar-nav-bs.html", "sidebar-ethical-ads.html"],
    "index": [],
    "get_started": [],
    "search": [],
}

# -- Options for the copybutton extension
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
