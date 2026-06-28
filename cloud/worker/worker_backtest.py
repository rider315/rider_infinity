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

import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

# When running as a cloud worker the project root is added to sys.path
# by the bootstrap script. When imported locally the project root is
# already on the path. Either way, strategies and utils must be importable.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from strategies import get_strategy
from strategies.base import SignalAction


# ── pip helpers (inlined to avoid importing utils which may pull MT5) ──

def _pips_to_price(pips: float, symbol: str, point: float) -> float:
    """Convert pips to price delta."""
    if "JPY" in symbol.upper():
        return pips * 0.01
    return pips * 0.0001


def _price_to_pips(price_delta: float, symbol: str, point: float) -> float:
    """Convert price delta to pips."""
    if "JPY" in symbol.upper():
        return price_delta / 0.01
    return price_delta / 0.0001


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

def check_sl_tp(trade: dict, bar: pd.Series) -> tuple[float, str] | None:
    """Check if a trade's SL or TP was hit on this bar."""
    sl, tp = trade["sl_price"], trade["tp_price"]
    high, low = bar["high"], bar["low"]
    if trade["direction"] == "BUY":
        sl_hit = low <= sl
        tp_hit = high >= tp
        if sl_hit and tp_hit:
            return (sl, "SL")
        if sl_hit:
            return (sl, "SL")
        if tp_hit:
            return (tp, "TP")
    else:
        sl_hit = high >= sl
        tp_hit = low <= tp
        if sl_hit and tp_hit:
            return (sl, "SL")
        if sl_hit:
            return (sl, "SL")
        if tp_hit:
            return (tp, "TP")
    return None


def calc_pnl_pips(direction: str, entry: float, exit_p: float,
                  symbol: str, point: float) -> float:
    """Calculate P&L in pips for a closed trade."""
    delta = (exit_p - entry) if direction == "BUY" else (entry - exit_p)
    return _price_to_pips(delta, symbol, point)


def compute_metrics(trades: list[_Trade], initial_balance: float,
                    final_equity: float) -> dict:
    """Compute performance metrics from a list of closed trades."""
    total = len(trades)
    if total == 0:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0,
            "total_pnl_pips": 0, "total_pnl_money": 0,
            "profit_factor": 0, "max_drawdown_pct": 0,
            "sharpe_ratio": 0, "avg_win_pips": 0, "avg_loss_pips": 0,
        }

    wins = sum(1 for t in trades if t.pnl_pips > 0)
    losses = total - wins
    win_rate = wins / total * 100

    total_pnl_pips = sum(t.pnl_pips for t in trades)
    total_pnl_money = round(final_equity - initial_balance, 2)

    gross_profit = sum(t.pnl_money for t in trades if t.pnl_money > 0)
    gross_loss = abs(sum(t.pnl_money for t in trades if t.pnl_money < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else (
        999.99 if gross_profit > 0 else 0
    )

    w_pips = [t.pnl_pips for t in trades if t.pnl_pips > 0]
    l_pips = [t.pnl_pips for t in trades if t.pnl_pips <= 0]
    avg_win = (sum(w_pips) / len(w_pips)) if w_pips else 0
    avg_loss = (sum(l_pips) / len(l_pips)) if l_pips else 0

    equity = initial_balance
    peak = equity
    max_dd_pct = 0.0
    for t in trades:
        equity += t.pnl_money
        if equity > peak:
            peak = equity
        dd_pct = ((peak - equity) / peak * 100) if peak > 0 else 0
        if dd_pct > max_dd_pct:
            max_dd_pct = dd_pct

    sharpe = 0.0
    if len(trades) > 1:
        rets = [t.pnl_money for t in trades]
        avg_r = sum(rets) / len(rets)
        var = sum((r - avg_r) ** 2 for r in rets) / (len(rets) - 1)
        std = math.sqrt(var) if var > 0 else 0
        sharpe = (avg_r / std * math.sqrt(252)) if std > 0 else 0

    return {
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 1),
        "total_pnl_pips": round(total_pnl_pips, 1),
        "total_pnl_money": round(total_pnl_money, 2),
        "profit_factor": round(min(profit_factor, 999.99), 2),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "sharpe_ratio": round(sharpe, 2),
        "avg_win_pips": round(avg_win, 1),
        "avg_loss_pips": round(avg_loss, 1),
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
