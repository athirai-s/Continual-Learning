from pathlib import Path
import sys

import pytest


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
