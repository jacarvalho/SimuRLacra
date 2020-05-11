# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# ----------
# Path setup
# ----------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

from pyrado import VERSION

# Add packages to the path
sys.path.insert(0, os.path.abspath('..'))  # doc folder
sys.path.insert(0, os.path.abspath('../..'))  # Pyrado folder

# -------------------
# Project information
# -------------------
project = 'Pyrado'
version = '.'.join(VERSION.split('.'))  # short version
release = VERSION  # full version including tags
copyright = '2020'
author = 'Fabio Muratore & Felix Treede & Robin Menzenbach'

# ---------------------
# General configuration
# ---------------------
# Add any Sphinx extension module names here, as strings
# They can be extensions coming with Sphinx (named 'sphinx.ext.*') or your custom ones
extensions = [
    'sphinx.ext.inheritance_diagram',
    'sphinx_math_dollar',
    'sphinx.ext.mathjax',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.viewcode',
    'sphinx_removed_in'
]


# Add any paths that contain templates here, relative to this directory
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and dirs to ignore when looking for source files
# This pattern also affects html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use
pygments_style = 'sphinx'

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'restructuredtext',
    '.md': 'markdown',
}

# -----------------------
# Options for HTML output
# -----------------------
# The theme to use for HTML and HTML Help pages. See the documentation for a list of builtin themes
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here, relative to this directory.
# They are copied after the builtin static files, so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

add_module_names = False