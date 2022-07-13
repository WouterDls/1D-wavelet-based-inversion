# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# add the project root to the python path
sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('../'))


# -- Project information -----------------------------------------------------

project = 'wabi'
copyright = '2022, Wouter Deleersnyder, Robin Thibaut'
author = 'Wouter Deleersnyder, Robin Thibaut'

# The full version, including alpha/beta/rc tags
release = '0.1.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx_rtd_theme",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "nbsphinx",
    "matplotlib.sphinxext.plot_directive",
    "IPython.sphinxext.ipython_directive",
    "IPython.sphinxext.ipython_console_highlighting",
    "sphinx.ext.inheritance_diagram",
]


autodoc_default_options = {'members': True,
                           'undoc-members': True,
                           'private-members': True,
                           'special-members': '__init__, __call__',
                           'inherited-members': False,
                           'show-inheritance': False}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

source_suffix = ['.rst', '.md']

add_function_parentheses = False
pygments_style = "sphinx"
html_domain_indices = False
html_use_index = False
html_copy_source = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages. See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"
# html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']

# cd = "cd docs"
# mc = "make clean"
# mh = "make html"
# auto = "sphinx-apidoc -f -o source/ ../skbel/"
#
# os.system(cd)
# os.system(mc)
# os.system(mh)
# os.system(auto)
