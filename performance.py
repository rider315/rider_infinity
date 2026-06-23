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
    "id", "bot_id", "ticket", "symbol", "strategy", "action", "lot",
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
        "by_strategy": _group_metrics(closed, "strategy"),
        "by_symbol": _group_metrics(closed, "symbol"),
        "by_exit_reason": _group_metrics(closed, "exit_reason"),
    }
