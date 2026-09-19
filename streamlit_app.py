"""Streamlit entry point: ``uv run streamlit run streamlit_app.py``.

Streamlit Community Cloud looks for this filename by default. The interface itself lives in
``citara.ui.app`` so that it is importable, typed and tested like the rest of the package.

The path line matters for deployment: Community Cloud installs ``requirements.txt`` and runs
this file from the repository root, and that installs the dependencies without installing
this project, whose code sits under ``src/``. Locally the package is installed already and
this changes nothing.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

# Imported after the path line above, which is why it is not at the top of the file.
from citara.ui.app import main

main()
