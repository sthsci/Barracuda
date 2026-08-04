from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "smc: opt-in smoke tests that execute PyMC SMC inference",
    )
