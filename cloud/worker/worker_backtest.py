"""
worker_backtest.py — Pure backtest functions extracted from StrategyOptimizer.

These functions run a single parameter-combo backtest against a
pre-fetched DataFrame. They have NO dependency on MetaTrader5, Flask,
or any local-only module — only pandas, math, and the strategy code.

Used by:
  - StrategyOptimizer (local, calls these instead of duplicating logic)
  - Cloud workers (EC2 instances running distributed optimisation)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

# When running as a cloud worker the project root is added to sys.path
# by the bootstrap script. When imported locally the project root is
# already on the path. Either way, ``strategies``, ``backtest_core``
# and ``instruments`` must be importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backtest_core import (
    calc_pnl_pips,
    check_sl_tp,
    equity_curve_from_trades,
    max_drawdown,
    pips_to_price as _pips_to_price,
    price_to_pips as _price_to_pips,
    profit_factor,
    sharpe_ratio,
)
from strategies import get_strategy
from strategies.base import SignalAction


# ── pip helpers (inlined to avoid importing utils which may pull MT5) ──

# ── Lightweight trade record (avoids importing full BacktestTrade) ────

class _Trade:
    __slots__ = ("pnl_pips", "pnl_money", "exit_reason", "entry_time", "exit_time")

    def __init__(self, pnl_pips: float, pnl_money: float, exit_reason: str,
                 entry_time: str, exit_time: str):
        self.pnl_pips = pnl_pips
        self.pnl_money = pnl_money
        self.exit_reason = exit_reason
        self.entry_time = entry_time
        self.exit_time = exit_time


# =====================================================================
#  Core backtest functions (stateless, no MT5)
# =====================================================================

def compute_metrics(trades: list[_Trade], initial_balance: float,
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
) -> dict:
    """Run a single backtest with the given params. Returns metrics dict.

    This is a pure function — no MT5, no database, no side effects.
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
    closed_trades: list[_Trade] = []
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
                    sl_p = round(entry_price - _pips_to_price(sl_pips, symbol, point), digits)
                    tp_p = round(entry_price + _pips_to_price(tp_pips, symbol, point), digits)
                    direction = "BUY"
                else:
                    sl_p = round(entry_price + _pips_to_price(sl_pips, symbol, point), digits)
                    tp_p = round(entry_price - _pips_to_price(tp_pips, symbol, point), digits)
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
                pnl_money = pnl_p * pip_value_per_lot * lot_size
                equity += pnl_money
                closed_trades.append(_Trade(
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
            pnl_money = pnl_p * pip_value_per_lot * lot_size
            equity += pnl_money
            closed_trades.append(_Trade(
                pnl_pips=round(pnl_p, 1),
                pnl_money=round(pnl_money, 2),
                exit_reason="END",
                entry_time=trade["entry_time"],
                exit_time=final_time,
            ))

    return compute_metrics(closed_trades, initial_balance, equity)
