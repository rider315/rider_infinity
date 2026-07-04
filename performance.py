"""
performance.py — Trade performance analytics & export helpers.

Turns the raw per-trade rows captured in ``data/bot_history.db`` (see
``bot_history.BotHistoryDB.get_trades``) into:

* a flat CSV of every trade (open + close detail), and
* a JSON summary of performance metrics with daily / weekly time-slab
  breakdowns plus per-strategy / per-symbol / per-exit-reason splits.

These two outputs are what you download from the dashboard and hand back
for optimisation.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime
from typing import Any, Dict, List, Optional

# Column order for the exported CSV (stable, analysis-friendly).
CSV_COLUMNS = [
    "id", "bot_id", "ticket", "symbol", "strategy", "logic_version",
    "action", "lot",
    "entry_price", "sl", "tp", "exit_price", "pips", "profit", "swap",
    "commission", "open_time", "close_time", "duration_sec",
    "exit_reason", "reason", "confidence", "status",
]


def trades_to_csv(trades: List[Dict[str, Any]]) -> str:
    """Render trade rows as a CSV string with a stable column order."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for t in trades:
        writer.writerow(t)
    return buf.getvalue()


def _to_perf_trade(run: Dict[str, Any], t: Dict[str, Any]) -> Dict[str, Any]:
    """Map a stored backtest-trade row onto the live-trade schema used here."""
    duration_sec = None
    try:
        et = datetime.fromisoformat(t.get("entry_time", ""))
        xt = datetime.fromisoformat(t.get("exit_time", ""))
        duration_sec = int((xt - et).total_seconds())
    except (ValueError, TypeError):
        pass
    return {
        "status": "CLOSED",
        "profit": t.get("pnl_money"),
        "pips": t.get("pnl_pips"),
        "open_time": t.get("entry_time", ""),
        "close_time": t.get("exit_time", ""),
        "duration_sec": duration_sec,
        "strategy": run.get("strategy_name", ""),
        "symbol": run.get("symbol", ""),
        "action": t.get("direction", ""),
        "exit_reason": t.get("exit_reason", ""),
        "confidence": t.get("signal_confidence"),
        "entry_price": t.get("entry_price"),
        "exit_price": t.get("exit_price"),
        "sl": t.get("sl_price"),
        "tp": t.get("tp_price"),
        "lot": t.get("lot_size"),
    }


def build_backtest_summary(run: Dict[str, Any], trades: List[Dict[str, Any]],
                           slab: str = "daily") -> Dict[str, Any]:
    """Performance summary for a backtest run (overall + time-of-day + splits).

    Reuses the same metrics engine as live trades so the analysis is identical,
    and adds the run's own metadata (symbol, timeframe, date range).
    """
    mapped = [_to_perf_trade(run, t) for t in trades]
    summary = build_summary(mapped, slab=slab)
    summary["backtest"] = {
        "run_id": run.get("run_id") or run.get("id"),
        "strategy": run.get("strategy_name"),
        "symbol": run.get("symbol"),
        "timeframe": run.get("timeframe"),
        "date_from": run.get("date_from"),
        "date_to": run.get("date_to"),
        "initial_balance": run.get("initial_balance"),
        "lot_size": run.get("lot_size"),
        "sl_pips": run.get("sl_pips"),
        "tp_pips": run.get("tp_pips"),
    }
    return summary


# Column order for the backtest CSV — entry/exit time first for easy analysis.
BACKTEST_CSV_COLUMNS = [
    "entry_time", "entry_hour_utc", "session", "exit_time", "direction",
    "symbol", "entry_price", "exit_price", "sl_price", "tp_price", "lot_size",
    "pnl_pips", "pnl_money", "exit_reason", "signal_confidence", "signal_reason",
]


def backtest_trades_to_csv(run: Dict[str, Any], trades: List[Dict[str, Any]]) -> str:
    """Render backtest trades as CSV, with entry-hour + session columns added."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=BACKTEST_CSV_COLUMNS,
                            extrasaction="ignore")
    writer.writeheader()
    for t in trades:
        row = dict(t)
        row["symbol"] = run.get("symbol", "")
        hour = _hour_of(t.get("entry_time", ""))
        row["entry_hour_utc"] = "" if hour is None else f"{hour:02d}:00"
        row["session"] = "" if hour is None else _session_for_hour(hour)
        writer.writerow(row)
    return buf.getvalue()


def _metrics(closed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute core performance metrics for a list of CLOSED trades."""
    n = len(closed)
    if n == 0:
        return {
            "trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "net_profit": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
            "profit_factor": None, "avg_win": 0.0, "avg_loss": 0.0,
            "avg_profit": 0.0, "best": 0.0, "worst": 0.0,
            "total_pips": 0.0, "max_drawdown": 0.0,
            "avg_duration_min": 0.0,
        }

    profits = [float(t.get("profit") or 0.0) for t in closed]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p < 0]
    gross_profit = sum(wins)
    gross_loss = abs(sum(losses))
    net = sum(profits)

    # Max drawdown from the running equity curve (trades are oldest-first).
    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in profits:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    durations = [t["duration_sec"] for t in closed if t.get("duration_sec")]
    pips = [float(t["pips"]) for t in closed if t.get("pips") is not None]

    return {
        "trades": n,
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / n * 100, 2),
        "net_profit": round(net, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "profit_factor": (round(gross_profit / gross_loss, 2)
                          if gross_loss > 0 else None),
        "avg_win": round(gross_profit / len(wins), 2) if wins else 0.0,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0.0,
        "avg_profit": round(net / n, 2),
        "best": round(max(profits), 2),
        "worst": round(min(profits), 2),
        "total_pips": round(sum(pips), 1),
        "max_drawdown": round(max_dd, 2),
        "avg_duration_min": (round(sum(durations) / len(durations) / 60, 1)
                             if durations else 0.0),
    }


def _group_metrics(closed: List[Dict[str, Any]], key: str) -> Dict[str, Any]:
    """Compute metrics grouped by a single field (strategy / symbol / reason)."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for t in closed:
        groups.setdefault(str(t.get(key) or "—"), []).append(t)
    return {k: _metrics(v) for k, v in sorted(groups.items())}


def _session_for_hour(hour: int) -> str:
    """Map a UTC hour to a forex trading session (where the action is)."""
    if 7 <= hour < 12:
        return "London"
    if 12 <= hour < 16:
        return "London/NY overlap"   # highest liquidity
    if 16 <= hour < 21:
        return "New York"
    return "Asian/Off-hours"          # 21:00-07:00 UTC, thin liquidity


def _hour_of(iso_time: str) -> Optional[int]:
    try:
        return datetime.fromisoformat(iso_time).hour
    except (ValueError, TypeError):
        return None


def build_time_of_day(closed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Break performance down by entry hour (UTC) and trading session.

    Answers "at what time should the bot trade?" — surfaces the hours/sessions
    where the edge actually lives so trading can be restricted to them.
    """
    by_hour: Dict[str, List[Dict[str, Any]]] = {}
    by_session: Dict[str, List[Dict[str, Any]]] = {}
    for t in closed:
        hour = _hour_of(t.get("open_time", ""))
        if hour is None:
            continue
        by_hour.setdefault(f"{hour:02d}:00", []).append(t)
        by_session.setdefault(_session_for_hour(hour), []).append(t)
    return {
        "by_hour_utc": {k: _metrics(v) for k, v in sorted(by_hour.items())},
        "by_session": {k: _metrics(v) for k, v in sorted(by_session.items())},
    }


def _slab_key(iso_time: str, slab: str) -> Optional[str]:
    """Bucket an ISO timestamp into a daily / weekly / monthly label."""
    try:
        dt = datetime.fromisoformat(iso_time)
    except (ValueError, TypeError):
        return None
    if slab == "weekly":
        iso = dt.isocalendar()
        return f"{iso[0]}-W{iso[1]:02d}"
    if slab == "monthly":
        return dt.strftime("%Y-%m")
    return dt.strftime("%Y-%m-%d")  # daily (default)


def build_summary(
    trades: List[Dict[str, Any]],
    slab: str = "daily",
    bot_id: Optional[int] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a full performance summary dict for the given trades.

    Open trades are reported as a count but excluded from profit metrics
    (they have no outcome yet).
    """
    closed = [t for t in trades if t.get("status") == "CLOSED"]
    open_count = sum(1 for t in trades if t.get("status") == "OPEN")

    # Time-slab buckets, keyed by close_time (when the outcome was realised).
    slabs: Dict[str, List[Dict[str, Any]]] = {}
    for t in closed:
        label = _slab_key(t.get("close_time") or t.get("open_time"), slab)
        if label:
            slabs.setdefault(label, []).append(t)

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "filters": {
            "bot_id": bot_id, "start": start, "end": end, "slab": slab,
        },
        "open_trades": open_count,
        "overall": _metrics(closed),
        "by_time_slab": {k: _metrics(v) for k, v in sorted(slabs.items())},
        "time_of_day": build_time_of_day(closed),
        "by_strategy": _group_metrics(closed, "strategy"),
        "by_symbol": _group_metrics(closed, "symbol"),
        "by_exit_reason": _group_metrics(closed, "exit_reason"),
        # Split by the code/settings fingerprint stamped at entry, so the
        # effect of a strategy change is directly comparable: trades under
        # the old logic vs trades under the new logic. "—" = trades from
        # before fingerprinting existed (or MT5 backfills).
        "by_logic_version": _group_metrics(closed, "logic_version"),
        "logic_version_info": _logic_version_info(closed),
    }


def _logic_version_info(closed: List[Dict[str, Any]]) -> Dict[str, Any]:
    """First-seen timestamps / strategy for each version present in the data.

    Makes the JSON export self-describing: a version hash can be ordered in
    time and traced to its settings snapshot in ``data/logic_versions.json``.
    """
    present = {str(t.get("logic_version") or "") for t in closed} - {""}
    if not present:
        return {}
    try:
        from logic_version import load_registry

        registry = load_registry()
    except Exception:
        return {}
    return {
        v: {
            "strategy": registry[v].get("strategy", ""),
            "first_seen": registry[v].get("first_seen", ""),
        }
        for v in sorted(present)
        if v in registry
    }
