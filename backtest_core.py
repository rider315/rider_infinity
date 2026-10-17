"""
backtest_core.py — The shared simulation kernel.

Bar-level exit detection, P&L conversion and the performance statistics
that every backtest path needs: the in-process engine
(:mod:`backtester`), the parameter sweep (:mod:`optimizer`), the
standalone cloud worker (``cloud/worker/worker_backtest.py``) and the
A/B script (:mod:`ab_backtest`).

Why this module exists
----------------------
These functions were copied into three files.  ``check_sl_tp`` and
``calc_pnl_pips`` were verbatim duplicates, the profit-factor and Sharpe
formulas were near-duplicates, and the cloud worker carried its own
hard-coded pip arithmetic that disagreed with the rest of the code-base
by 1000× on gold.  Worse, the dependency ran the wrong way: the core
``optimizer`` imported its kernel from ``cloud.worker``, so the local
sweep could not run without the cloud package.

Everything here is pure: no MT5, no network, no I/O — only the standard
library, pandas and :mod:`instruments`.  That keeps it importable from
the cloud worker (which ships as a tarball with no project packaging)
and trivially unit-testable.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional, Sequence

import pandas as pd

from instruments import pip_size, pip_value_per_lot, pips_to_price, price_to_pips

__all__ = [
    "SQRT_TRADING_DAYS",
    "calc_pnl_money",
    "calc_pnl_pips",
    "check_sl_tp",
    "equity_curve_from_trades",
    "max_consecutive",
    "max_drawdown",
    "pip_size",
    "pip_value_per_lot",
    "pips_to_price",
    "price_to_pips",
    "profit_factor",
    "sharpe_ratio",
]

#: Annualisation factor applied to per-trade returns in
#: :func:`sharpe_ratio`.  Treating each trade as one "day" is a
#: simplification, but it is the convention this project has always used
#: and keeps historical runs comparable.
SQRT_TRADING_DAYS = math.sqrt(252)

#: Ceiling reported instead of infinity when there are no losing trades,
#: so the value stays JSON-serialisable and sortable.
_PROFIT_FACTOR_CAP = 999.99


# ---------------------------------------------------------------------
# Bar-level exit detection
# ---------------------------------------------------------------------

def check_sl_tp(
    trade: dict, bar: Any, *, sl_key: str = "sl_price",
    tp_key: str = "tp_price",
) -> Optional[tuple[float, str]]:
    """Return ``(exit_price, reason)`` if *bar* triggers *trade*'s SL or TP.

    Parameters
    ----------
    trade
        Open trade with ``direction`` (``"BUY"``/``"SELL"``) and the stop
        and target prices under *sl_key* / *tp_key*.
    bar
        A row with ``high`` and ``low``.
    sl_key, tp_key
        Names of the stop/target keys in *trade*.

    Returns
    -------
    tuple[float, str] | None
        ``(price, "SL")`` or ``(price, "TP")``, or ``None`` if neither
        level was touched.

    Notes
    -----
    When a single bar spans both levels, the stop is assumed to have been
    hit first.  Intrabar order is unknowable from OHLC data, so the
    pessimistic choice is the only honest one — assuming the target would
    inflate every result and the error compounds across thousands of
    trades.
    """
    sl = trade[sl_key]
    tp = trade[tp_key]
    high = bar["high"]
    low = bar["low"]

    if trade["direction"] == "BUY":
        sl_hit = low <= sl
        tp_hit = high >= tp
    else:
        sl_hit = high >= sl
        tp_hit = low <= tp

    if sl_hit:
        return (sl, "SL")   # pessimistic when both are touched
    if tp_hit:
        return (tp, "TP")
    return None


# ---------------------------------------------------------------------
# P&L
# ---------------------------------------------------------------------

def calc_pnl_pips(
    direction: str, entry_price: float, exit_price: float,
    symbol: str, point: float,
) -> float:
    """Return a closed trade's P&L in pips (negative for a loss)."""
    delta = (
        exit_price - entry_price if direction == "BUY"
        else entry_price - exit_price
    )
    return price_to_pips(delta, symbol, point)


def calc_pnl_money(
    pnl_pips: float, lot: float, pip_value: float,
    spread_pips: float = 0.0, commission: float = 0.0,
) -> float:
    """Convert a pip P&L into account currency.

    Parameters
    ----------
    pnl_pips
        Gross P&L in pips.
    lot
        Trade volume.
    pip_value
        Account-currency value of one pip per lot
        (:func:`instruments.pip_value_per_lot`).
    spread_pips
        Round-trip spread cost, charged against the trade.
    commission
        Flat commission, charged against the trade.
    """
    return (pnl_pips - spread_pips) * lot * pip_value - commission


# ---------------------------------------------------------------------
# Performance statistics
# ---------------------------------------------------------------------

def profit_factor(
    pnls: Iterable[float], cap: float = _PROFIT_FACTOR_CAP,
) -> float:
    """Gross profit divided by gross loss.

    Returns *cap* when there are winners but no losers (rather than
    infinity, which does not survive JSON), and ``0.0`` when there are no
    winners.
    """
    pnls = list(pnls)
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    if gross_loss > 0:
        return min(gross_profit / gross_loss, cap)
    return cap if gross_profit > 0 else 0.0


def sharpe_ratio(pnls: Sequence[float]) -> float:
    """Annualised Sharpe ratio over per-trade returns.

    Uses the sample standard deviation (``n − 1``), and returns ``0.0``
    for fewer than two trades or zero variance.
    """
    if len(pnls) < 2:
        return 0.0
    mean = sum(pnls) / len(pnls)
    variance = sum((p - mean) ** 2 for p in pnls) / (len(pnls) - 1)
    if variance <= 0:
        return 0.0
    return mean / math.sqrt(variance) * SQRT_TRADING_DAYS


def equity_curve_from_trades(
    pnls: Iterable[float], initial_balance: float,
) -> list[float]:
    """Return the running equity after each trade, starting at the balance."""
    equity = initial_balance
    curve = [equity]
    for pnl in pnls:
        equity += pnl
        curve.append(equity)
    return curve


def max_drawdown(equity: Sequence[float]) -> tuple[float, float]:
    """Return ``(max_drawdown_money, max_drawdown_pct)`` for an equity curve.

    The percentage is measured against the running peak, which is the
    figure a trader actually experiences.
    """
    if not equity:
        return (0.0, 0.0)
    peak = equity[0]
    worst_money = 0.0
    worst_pct = 0.0
    for value in equity:
        if value > peak:
            peak = value
        drop = peak - value
        pct = (drop / peak * 100) if peak > 0 else 0.0
        worst_money = max(worst_money, drop)
        worst_pct = max(worst_pct, pct)
    return (worst_money, worst_pct)


def max_consecutive(outcomes: Iterable[bool], target: bool = True) -> int:
    """Longest run of *target* in *outcomes* (e.g. consecutive wins)."""
    best = run = 0
    for outcome in outcomes:
        if outcome is target or outcome == target:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best
