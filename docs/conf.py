# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'SLDL'
copyright = '2023, Nikita Pavlichenko'
author = 'Nikita Pavlichenko'
release = '0.0.2'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ["custom.css"]

add_module_names = False
autodoc_member_order = 'bysource'

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy', None),
    'torch': ('https://pytorch.org/docs/master', None),
    'PIL': ('https://pillow.readthedocs.io/en/stable/', None),
}
