# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'XLB'
copyright = '2025, Medyan Naser'
author = 'Medyan Naser'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",            # for Markdown support
    "sphinx.ext.autodoc",     # for docstrings
    "sphinx.ext.napoleon",    # for Google-style/Numpy-style docstrings
    "sphinx_book_theme",
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_static_path = ['_static']

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/Autodesk/XLB.git",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
    "use_fullscreen_button": True,
    "navigation_with_keys": True,
    "show_navbar_depth": 2,
}

extensions.append("myst_parser")
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

root_doc = 'index'
