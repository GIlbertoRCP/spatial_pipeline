import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.abspath("../src"))

project = "spatial_pipeline"
author = "Gilberto Romero-Cano"
copyright = f"{datetime.now().year}, {author}"
release = "0.4.0"

extensions = [
    "myst_nb",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
]

# MyST-NB settings
nb_execution_mode = "off"  # Render notebooks statically without re-executing during build
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "amsmath",
    "fieldlist",
]

# Autodoc settings
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# HTML output settings
html_theme = "sphinx_book_theme"
html_title = "spatial_pipeline Documentation"
html_theme_options = {
    "repository_url": "https://github.com/gilbertorcp/spatial_pipeline",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "path_to_docs": "docs",
    "show_navbar_depth": 2,
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "myst-nb",
    ".ipynb": "myst-nb",
}

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
