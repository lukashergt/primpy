# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import sys
import os
sys.path.append(os.path.abspath('../'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'primpy'
copyright = '2022, Lukas Hergt'
author = 'Lukas Hergt'

version = {}
with open("../../primpy/__version__.py") as versionfile:
    exec(versionfile.read(), version)
version = version['__version__']
release = version


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.autosectionlabel',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.imgconverter',
    'sphinx_copybutton',
    'sphinx_autodoc_typehints',
    #'sphinx.ext.napoleon',
    'numpydoc',
    'matplotlib.sphinxext.plot_directive',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['../templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = ['.rst', '.md']
exclude_patterns = []

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'


# -- Options for autodoc -------------------------------------------------
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__call__',
    'undoc-members': False,
}
autosummary_generate = True


# -- Options for autosectionlabel------------------------------------------
autosectionlabel_prefix_document = True


# -- Options for sphinx-copybutton ----------------------------------------
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True


# -- Options for numpydoc -------------------------------------------------
numpydoc_class_members_toctree = False
numpydoc_show_inherited_class_members = False
#numpydoc_show_class_members = False 


# -- Options for matplotlib extension ----------------------------------------
plot_rcparams = {'savefig.bbox': 'tight'}
plot_apply_rcparams = True  # if context option is used
plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_pre_code = "import numpy as np; from matplotlib import pyplot as plt"


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

html_static_path = []


# -- Options for intersphinx extension ---------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy':('https://numpy.org/doc/stable/', None),
    'scipy':('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib':('https://matplotlib.org/stable/', None),
    'pyoscode': ('https://oscode.readthedocs.io/en/latest', None)
}
