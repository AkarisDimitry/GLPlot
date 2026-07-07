# Configuration file for the Sphinx documentation builder.
import os
import sys
from pathlib import Path

# Add project root to path for autodoc
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Project information
project = "GLPlot"
copyright = "2024, Juan Manuel Lombardi"
author = "Juan Manuel Lombardi"
release = "0.1.3"

# Sphinx extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinxcontrib.autodoc_pydantic",
]

# Autodoc options
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "undoc-members": True,
    "show-inheritance": True,
    "special-members": "__init__",
    "imported-members": False,
}

autosummary_generate = True
autosummary_generate_overwrite = False

# Napoleon (Google-style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_param = True
napoleon_use_rtype = True

# Intersphinx mapping
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
}

# HTML theme
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": False,
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "vcs_pageview_mode": "view",
    "style_nav_header_background": "#2c3e50",
}

# HTML static files
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_logo = None  # Set to logo path if available
html_favicon = None

# Source files
source_suffix = {".rst": None}
master_doc = "index"

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Pygments highlighting
pygments_style = "sphinx"
highlight_language = "python"

# Ensure proper handling of type hints
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# ViewCode settings
viewcode_follow_imported_members = True
