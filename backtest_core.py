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
from strategies import get_strategy
from strategies.base import SignalAction

__all__ = [
    "SQRT_TRADING_DAYS",
    "Trade",
    "compute_metrics",
    "run_single",
    "spread_pips_from_symbol",
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


def spread_pips_from_symbol(
    broker_spread_points: float, symbol: str, point: float,
    max_spread_points: float = 30.0,
) -> float:
    """Per-trade spread cost in pips, from the broker's spread snapshot.

    Historical spread is not in candle data, so a backtest charges the
    broker's *current* spread — capped at ``max_spread_points``, because
    ``symbol_info().spread`` is a live snapshot and brokers blow it out on
    a closed market (weekend gold has been seen at 640 points). Live
    trading never opens a position above the ``MAX_SPREAD_POINTS`` gate in
    :class:`~risk_manager.RiskManager`, so the simulation must not charge
    more than that either.

    Returns ``0.0`` when the point size or the snapshot is unusable, which
    makes the backtest cost-free — optimistic, so callers should prefer a
    sane default over passing nothing.
    """
    size = pip_size(symbol, point)
    if size <= 0 or not broker_spread_points:
        return 0.0
    points_per_pip = size / point if point > 0 else 10.0
    capped = min(float(broker_spread_points), float(max_spread_points or 30.0))
    return capped / points_per_pip


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


class Trade:
    """A closed trade, as the simulation loop records it.

    Deliberately minimal — ``__slots__``, no validation — because a
    parameter sweep builds hundreds of thousands of these. The richer
    :class:`backtester.BacktestTrade` is for results that get persisted.
    """


    __slots__ = ("pnl_pips", "pnl_money", "exit_reason", "entry_time", "exit_time")

    def __init__(self, pnl_pips: float, pnl_money: float, exit_reason: str,
                 entry_time: str, exit_time: str):
        self.pnl_pips = pnl_pips
        self.pnl_money = pnl_money
        self.exit_reason = exit_reason
        self.entry_time = entry_time
        self.exit_time = exit_time


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def compute_metrics(trades: list[Trade], initial_balance: float,
                    final_equity: float) -> dict:
    """Compute performance metrics from a list of closed trades.

    The statistics themselves come from :mod:`backtest_core` so the
    cloud worker reports exactly the figures the in-process engine does.
    """
    total = len(trades)
    if total == 0:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "total_pnl_pips": 0, "total_pnl_money": 0,
            "profit_factor": 0, "max_drawdown_pct": 0,
            "sharpe_ratio": 0, "avg_win_pips": 0, "avg_loss_pips": 0,
        }

    wins = sum(1 for t in trades if t.pnl_pips > 0)
    money = [t.pnl_money for t in trades]

    win_pips = [t.pnl_pips for t in trades if t.pnl_pips > 0]
    loss_pips = [t.pnl_pips for t in trades if t.pnl_pips <= 0]

    _, max_dd_pct = max_drawdown(
        equity_curve_from_trades(money, initial_balance)
    )

    return {
        "total_trades": total,
        "wins": wins,
        "losses": total - wins,
        "win_rate": round(wins / total * 100, 1),
        "total_pnl_pips": round(sum(t.pnl_pips for t in trades), 1),
        "total_pnl_money": round(final_equity - initial_balance, 2),
        "profit_factor": round(profit_factor(money), 2),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "sharpe_ratio": round(sharpe_ratio(money), 2),
        "avg_win_pips": round(
            sum(win_pips) / len(win_pips) if win_pips else 0, 1
        ),
        "avg_loss_pips": round(
            sum(loss_pips) / len(loss_pips) if loss_pips else 0, 1
        ),
    }


# ---------------------------------------------------------------------
# The simulation loop
# ---------------------------------------------------------------------

def run_single(
    strategy_name: str,
    params: dict,
    df: pd.DataFrame,
    symbol: str,
    point: float,
    digits: int,
    pip_value_per_lot: float,
    lot_size: float,
    initial_balance: float,
    default_sl: float,
    default_tp: float,
    max_open_trades: int = 5,
    spread_pips: float = 0.0,
    commission_per_trade: float = 0.0,
) -> dict:
    """Run a single backtest with the given params. Returns metrics dict.

    This is a pure function — no MT5, no database, no side effects.

    Parameters
    ----------
    spread_pips
        Per-trade spread cost, charged against every closed trade.
        Derive it with :func:`spread_pips_from_symbol`.

        **Pass it.** The default of 0 makes the run cost-free, which is
        how this kernel used to behave unconditionally while
        ``BacktestEngine._execute`` charged the spread — so the
        parameter sweep and the cloud workers ranked every parameter set
        on gross P&L. On a 1.5-pip EURUSD spread that overstated a
        typical run's profit by 20-50%, and more for a scalper whose
        target is a few pips, so the "best" parameters it saved were
        chosen partly for trading more often.
    commission_per_trade
        Flat commission charged against every closed trade.
    """
    indicator_params = {k: v for k, v in params.items()
                        if k not in ("sl_pips", "tp_pips")}
    sl_pips = params.get("sl_pips", default_sl)
    tp_pips = params.get("tp_pips", default_tp)

    strategy = get_strategy(strategy_name, sl_pips=sl_pips, tp_pips=tp_pips,
                            **indicator_params)
    max_hold_bars = int(getattr(strategy, "max_hold_bars", 0) or 0)

    total_bars = len(df)
    WARMUP = min(200, total_bars // 3)
    equity = initial_balance
    open_trades: list[dict] = []
    closed_trades: list[Trade] = []
    pending_signal = None
    trade_counter = 0

    for i in range(WARMUP, total_bars):
        bar = df.iloc[i]
        bar_time = bar["time"].isoformat() if hasattr(bar["time"], "isoformat") else str(bar["time"])

        # Execute pending signal at this bar's open
        if pending_signal is not None:
            sig = pending_signal
            pending_signal = None

            if len(open_trades) < max_open_trades:
                entry_price = bar["open"]
                trade_counter += 1

                if sig.action == SignalAction.BUY:
                    sl_p = round(entry_price - pips_to_price(sl_pips, symbol, point), digits)
                    tp_p = round(entry_price + pips_to_price(tp_pips, symbol, point), digits)
                    direction = "BUY"
                else:
                    sl_p = round(entry_price + pips_to_price(sl_pips, symbol, point), digits)
                    tp_p = round(entry_price - pips_to_price(tp_pips, symbol, point), digits)
                    direction = "SELL"

                open_trades.append({
                    "direction": direction,
                    "entry_price": entry_price,
                    "entry_time": bar_time,
                    "entry_index": i,
                    "sl_price": sl_p, "tp_price": tp_p,
                })

        # Check SL/TP for open trades
        still_open = []
        for trade in open_trades:
            hit = check_sl_tp(trade, bar)
            if not hit and max_hold_bars and (i - trade["entry_index"]) >= max_hold_bars:
                hit = (bar["close"], "TIME")
            if hit:
                exit_price, exit_reason = hit
                pnl_p = calc_pnl_pips(trade["direction"], trade["entry_price"],
                                      exit_price, symbol, point)
                pnl_money = calc_pnl_money(
                    pnl_p, lot_size, pip_value_per_lot,
                    spread_pips=spread_pips,
                    commission=commission_per_trade,
                )
                equity += pnl_money
                closed_trades.append(Trade(
                    pnl_pips=round(pnl_p, 1),
                    pnl_money=round(pnl_money, 2),
                    exit_reason=exit_reason,
                    entry_time=trade["entry_time"],
                    exit_time=bar_time,
                ))
            else:
                still_open.append(trade)
        open_trades = still_open

        # Run strategy
        window = df.iloc[max(0, i - 499):i + 1].copy().reset_index(drop=True)
        try:
            signal = strategy.analyze(window, symbol)
        except Exception:
            signal = None

        if signal and signal.action in (SignalAction.BUY, SignalAction.SELL):
            pending_signal = signal

    # Close remaining trades at final bar
    if open_trades:
        final = df.iloc[-1]
        final_time = final["time"].isoformat() if hasattr(final["time"], "isoformat") else str(final["time"])
        for trade in open_trades:
            pnl_p = calc_pnl_pips(trade["direction"], trade["entry_price"],
                                  final["close"], symbol, point)
            pnl_money = calc_pnl_money(
                pnl_p, lot_size, pip_value_per_lot,
                spread_pips=spread_pips,
                commission=commission_per_trade,
            )
            equity += pnl_money
            closed_trades.append(Trade(
                pnl_pips=round(pnl_p, 1),
                pnl_money=round(pnl_money, 2),
                exit_reason="END",
                entry_time=trade["entry_time"],
                exit_time=final_time,
            ))

    return compute_metrics(closed_trades, initial_balance, equity)
