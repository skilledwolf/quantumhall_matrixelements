from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
GALLERY_ROOT = ROOT / "docs" / "generated" / "gallery"

sys.path.insert(0, str(SRC))

matplotlib.use("Agg")
GALLERY_ROOT.mkdir(parents=True, exist_ok=True)
(GALLERY_ROOT / "images" / "thumb").mkdir(parents=True, exist_ok=True)

project = "quantumhall-matrixelements"
author = "Tobias Wolf, Sparsh Mishra"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_gallery.gen_gallery",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "examples/README.rst"]

myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "dollarmath",
]
myst_heading_anchors = 3

autodoc_preserve_defaults = True
autodoc_typehints = "description"
napoleon_google_docstring = False
napoleon_numpy_docstring = True

mathjax3_config = {
    "tex": {
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
    }
}

sphinx_gallery_conf = {
    "examples_dirs": "examples",
    "gallery_dirs": "generated/gallery",
    "filename_pattern": r"/plot_.*\.py",
    "download_all_examples": False,
    "remove_config_comments": True,
    "show_memory": False,
}

html_theme = "furo"
html_title = "quantumhall-matrixelements"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "source_repository": "https://github.com/skilledwolf/quantumhall_matrixelements/",
    "source_branch": "main",
    "source_directory": "docs/",
}
