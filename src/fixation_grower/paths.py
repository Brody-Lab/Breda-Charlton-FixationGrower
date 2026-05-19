"""Resolve data and figure directories.

Resolution order for data:
1. FIXATION_GROWER_DATA_DIR environment variable
2. <repo_root>/data/
"""

import os
from pathlib import Path

# Repo root is three levels up from this file: src/fixation_grower/paths.py
_REPO_ROOT = Path(__file__).parent.parent.parent


def data_dir() -> Path:
    env = os.environ.get("FIXATION_GROWER_DATA_DIR")
    if env:
        return Path(env)
    return _REPO_ROOT / "data"


def figures_dir() -> Path:
    return _REPO_ROOT / "figures"
