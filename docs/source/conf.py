"""Sphinx configuration.

The package is read straight from ``src/`` and every heavy third-party import
is mocked, so the documentation builds without installing torch, evaluate, or
the OpenAI SDK. That keeps the ReadTheDocs build small and fast.
"""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath("../../src"))

project = "llm-forensic-timeline"
author = "Hudan Studiawan, Frank Breitinger, Mark Scanlon"
copyright = f"{date.today().year}, {author}"
release = "0.0.2"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autodoc_mock_imports = [
    "dotenv",
    "evaluate",
    "modern_bert_score",
    "openai",
    "torch",
]

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "sphinx_rtd_theme"
html_static_path = []
