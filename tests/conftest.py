"""Pytest bootstrap for the Rider Infinity test-suite.

Two things have to happen *before* any project module is imported:

1. ``MetaTrader5`` must exist.  It is a Windows-only package, so on Linux
   or CI every project module that does ``import MetaTrader5 as mt5``
   would raise ``ModuleNotFoundError``.  We register ``tests/mt5_stub.py``
   under that name in ``sys.modules``.
2. The mandatory MT5 credentials must be present in the environment,
   because ``Config.__init__`` deliberately fails fast without them.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent

# Make both the project and the stub importable.
for path in (str(_PROJECT_ROOT), str(_TESTS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

# -- 1. Install the MetaTrader5 stub ---------------------------------
import mt5_stub  # noqa: E402

sys.modules.setdefault("MetaTrader5", mt5_stub)

# -- 2. Provide the credentials Config requires ----------------------
# Set before import so a developer's real .env can never leak into a
# test run (python-dotenv does not override existing env vars).
os.environ.setdefault("MT5_LOGIN", "1")
os.environ.setdefault("MT5_PASSWORD", "test-password")
os.environ.setdefault("MT5_SERVER", "Test-Server")

import pytest  # noqa: E402


@pytest.fixture
def mt5():
    """The MetaTrader5 stub, reset to its defaults after each test."""
    yield mt5_stub
    mt5_stub.reset()


@pytest.fixture
def config():
    """A fresh ``Config`` built from the test environment."""
    from config import Config

    return Config()
