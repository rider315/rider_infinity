"""
worker_backtest.py — Cloud-worker view of the backtest kernel.

Kept as a stable import path for the deployed worker (``worker.py`` does
``from cloud.worker.worker_backtest import run_single``) and for any
in-flight job whose package was built before this split.  Everything it
exposes now lives in :mod:`backtest_core`, which is where the local
engine and the parameter sweep read it from too.

Previously this file held its own copy of ``check_sl_tp``,
``calc_pnl_pips`` and ``compute_metrics``, plus its own hard-coded pip
arithmetic (``0.01`` for JPY, ``0.0001`` otherwise, ignoring the
symbol's real ``point``) — which read gold's pip as 0.0001 instead of
0.1, a factor of 1000. A cloud sweep and a local sweep could rank the
same parameters differently.
"""

from __future__ import annotations

import sys
from pathlib import Path

# When running as a cloud worker the project root is added to sys.path by
# the bootstrap script. When imported locally it is already there. Either
# way ``backtest_core``, ``instruments`` and ``strategies`` must resolve.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backtest_core import (  # noqa: E402,F401
    Trade as _Trade,
    calc_pnl_pips,
    check_sl_tp,
    compute_metrics,
    pips_to_price as _pips_to_price,
    price_to_pips as _price_to_pips,
    run_single,
)

__all__ = [
    "calc_pnl_pips",
    "check_sl_tp",
    "compute_metrics",
    "run_single",
]
