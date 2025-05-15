# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
import sphinx_rtd_theme

sys.path.insert(0, os.path.abspath('../../../')) # Goes up to find Drone_class1.py



project = 'MLFD Doc'
copyright = '2025, Corentin'
author = 'Corentin'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
]
autodoc_mock_imports = [
    "statsmodels", "pandas", "scipy", "sklearn", "sympy", "matplotlib","Profil",
    "Propeller", "Section", "Propeller_2", "Section_2","BEMT_NN_INVERSE_CORRECT", "Simulator_FUNCTIONS","mpl_toolkits"
]




templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
