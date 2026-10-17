"""
backtester.py — Strategy Backtester for the Rider Infinity Trading Platform.

Runs any registered strategy against historical MT5 data bar-by-bar,
simulates trades with SL/TP, tracks equity curves, computes performance
metrics, and persists results to SQLite for the dashboard to display.

Usage (standalone)::

    from backtester import BacktestEngine
    engine = BacktestEngine()
    result = engine.run("ema", "EURUSD", "H1", date_from, date_to)
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

import MetaTrader5 as mt5
import pandas as pd

from backtest_core import (
    calc_pnl_pips,
    check_sl_tp,
    max_consecutive,
    equity_curve_from_trades,
    max_drawdown,
    pip_size as _pip_size,
    pip_value_per_lot as _pip_value_per_lot,
    profit_factor,
    sharpe_ratio,
)
from config import Config, TIMEFRAME_MAP
from strategies import get_strategy
from strategies.base import SignalAction
from utils import pips_to_price, price_to_pips, setup_logger

logger = setup_logger("backtester")

# ── Paths ────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
_DB_PATH = _PROJECT_ROOT / "data" / "backtest_results.db"


# =====================================================================
#  Data Classes
# =====================================================================

@dataclass
class BacktestTrade:
    """A single simulated trade from a backtest run."""
    trade_id: int = 0
    entry_time: str = ""
    exit_time: str = ""
    direction: str = ""          # "BUY" or "SELL"
    entry_price: float = 0.0
    exit_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    lot_size: float = 0.0
    pnl_pips: float = 0.0
    pnl_money: float = 0.0
    exit_reason: str = ""        # "TP", "SL", "END" (end of data)
    signal_confidence: float = 0.0
    signal_reason: str = ""

    @property
    def is_win(self) -> bool:
        return self.pnl_pips > 0


@dataclass
class BacktestResult:
    """Complete results from a backtest run."""
    run_id: str = ""
    strategy_name: str = ""
    symbol: str = ""
    timeframe: str = ""
    date_from: str = ""
    date_to: str = ""
    initial_balance: float = 10000.0
    lot_size: float = 0.01
    sl_pips: float = 0.0
    tp_pips: float = 0.0
    strategy_params: dict = field(default_factory=dict)  # custom indicator params from optimizer
    status: str = "running"      # running | completed | failed
    error_msg: str = ""
    progress: float = 0.0       # 0.0 → 1.0

    # Trade data
    trades: list[BacktestTrade] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)

    # Computed metrics
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl_pips: float = 0.0
    total_pnl_money: float = 0.0
    profit_factor: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    avg_win_pips: float = 0.0
    avg_loss_pips: float = 0.0
    best_trade_pips: float = 0.0
    worst_trade_pips: float = 0.0
    sharpe_ratio: float = 0.0
    avg_trade_duration_mins: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    def compute_metrics(self) -> None:
        """Calculate all performance metrics from the trades list."""
        if not self.trades:
            return

        self.total_trades = len(self.trades)
        self.wins = sum(1 for t in self.trades if t.is_win)
        self.losses = self.total_trades - self.wins
        self.win_rate = (self.wins / self.total_trades * 100) if self.total_trades else 0

        self.total_pnl_pips = sum(t.pnl_pips for t in self.trades)
        self.total_pnl_money = sum(t.pnl_money for t in self.trades)

        # Profit factor (shared with the sweep and the cloud worker).
        self.profit_factor = profit_factor(
            [t.pnl_money for t in self.trades]
        )

        # Average win/loss
        winning_pips = [t.pnl_pips for t in self.trades if t.is_win]
        losing_pips = [t.pnl_pips for t in self.trades if not t.is_win]
        self.avg_win_pips = (sum(winning_pips) / len(winning_pips)) if winning_pips else 0
        self.avg_loss_pips = (sum(losing_pips) / len(losing_pips)) if losing_pips else 0

        # Best / worst
        all_pips = [t.pnl_pips for t in self.trades]
        self.best_trade_pips = max(all_pips) if all_pips else 0
        self.worst_trade_pips = min(all_pips) if all_pips else 0

        # Max drawdown. Prefer the recorded equity curve (it includes
        # open-position marks); fall back to reconstructing it from the
        # closed trades when no curve was captured.
        if self.equity_curve:
            equity = [pt["equity"] for pt in self.equity_curve]
        else:
            equity = equity_curve_from_trades(
                [t.pnl_money for t in self.trades], self.initial_balance,
            )
        self.max_drawdown, self.max_drawdown_pct = max_drawdown(equity)

        # Sharpe ratio (simplified: per-trade returns annualised).
        self.sharpe_ratio = sharpe_ratio(
            [t.pnl_money for t in self.trades]
        )

        # Average trade duration
        durations = []
        for t in self.trades:
            try:
                entry_dt = datetime.fromisoformat(t.entry_time)
                exit_dt = datetime.fromisoformat(t.exit_time)
                durations.append((exit_dt - entry_dt).total_seconds() / 60)
            except (ValueError, TypeError):
                pass
        self.avg_trade_duration_mins = (
            sum(durations) / len(durations)
        ) if durations else 0

        # Consecutive wins/losses
        self.max_consecutive_wins = _max_consecutive(self.trades, win=True)
        self.max_consecutive_losses = _max_consecutive(self.trades, win=False)

    def to_summary(self) -> dict:
        """Return a JSON-safe summary dict (without trade lists)."""
        return {
            "run_id": self.run_id,
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "initial_balance": self.initial_balance,
            "lot_size": self.lot_size,
            "sl_pips": self.sl_pips,
            "tp_pips": self.tp_pips,
            "status": self.status,
            "error_msg": self.error_msg,
            "progress": self.progress,
            "total_trades": self.total_trades,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": round(self.win_rate, 1),
            "total_pnl_pips": round(self.total_pnl_pips, 1),
            "total_pnl_money": round(self.total_pnl_money, 2),
            "profit_factor": round(self.profit_factor, 2),
            "max_drawdown": round(self.max_drawdown, 2),
            "max_drawdown_pct": round(self.max_drawdown_pct, 1),
            "avg_win_pips": round(self.avg_win_pips, 1),
            "avg_loss_pips": round(self.avg_loss_pips, 1),
            "best_trade_pips": round(self.best_trade_pips, 1),
            "worst_trade_pips": round(self.worst_trade_pips, 1),
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "avg_trade_duration_mins": round(self.avg_trade_duration_mins, 0),
            "max_consecutive_wins": self.max_consecutive_wins,
            "max_consecutive_losses": self.max_consecutive_losses,
        }


def _max_consecutive(trades: list[BacktestTrade], win: bool) -> int:
    """Longest run of winning (or losing) trades.

    Thin wrapper over :func:`backtest_core.max_consecutive`.
    """
    return max_consecutive((t.is_win for t in trades), target=win)


# =====================================================================
#  SQLite Persistence
# =====================================================================

class BacktestDB:
    """SQLite storage for backtest runs, trades, and equity curves."""

    def __init__(self, db_path: str | Path | None = None):
        self._db_path = str(db_path or _DB_PATH)
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_tables()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        # WAL lets the dashboard read run history while a backtest
        # thread is still writing, instead of blocking on the writer
        # (the other two stores in this project already use it).
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_tables(self) -> None:
        with self._lock, self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    date_from TEXT NOT NULL,
                    date_to TEXT NOT NULL,
                    initial_balance REAL DEFAULT 10000,
                    lot_size REAL DEFAULT 0.01,
                    sl_pips REAL DEFAULT 0,
                    tp_pips REAL DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    wins INTEGER DEFAULT 0,
                    losses INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    total_pnl_pips REAL DEFAULT 0,
                    total_pnl_money REAL DEFAULT 0,
                    profit_factor REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    max_drawdown_pct REAL DEFAULT 0,
                    sharpe_ratio REAL DEFAULT 0,
                    avg_win_pips REAL DEFAULT 0,
                    avg_loss_pips REAL DEFAULT 0,
                    best_trade_pips REAL DEFAULT 0,
                    worst_trade_pips REAL DEFAULT 0,
                    avg_trade_duration_mins REAL DEFAULT 0,
                    max_consecutive_wins INTEGER DEFAULT 0,
                    max_consecutive_losses INTEGER DEFAULT 0,
                    status TEXT DEFAULT 'running',
                    error_msg TEXT DEFAULT '',
                    progress REAL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS backtest_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
                    entry_time TEXT,
                    exit_time TEXT,
                    direction TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    sl_price REAL,
                    tp_price REAL,
                    lot_size REAL,
                    pnl_pips REAL,
                    pnl_money REAL,
                    exit_reason TEXT,
                    signal_confidence REAL,
                    signal_reason TEXT
                );

                CREATE TABLE IF NOT EXISTS backtest_equity (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL REFERENCES backtest_runs(id) ON DELETE CASCADE,
                    bar_time TEXT,
                    equity REAL
                );

                CREATE INDEX IF NOT EXISTS idx_trades_run ON backtest_trades(run_id);
                CREATE INDEX IF NOT EXISTS idx_equity_run ON backtest_equity(run_id);

                CREATE TABLE IF NOT EXISTS optimized_strategies (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    strategy_name TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    params TEXT NOT NULL DEFAULT '{}',
                    sl_pips REAL DEFAULT 0,
                    tp_pips REAL DEFAULT 0,
                    total_trades INTEGER DEFAULT 0,
                    win_rate REAL DEFAULT 0,
                    total_pnl_pips REAL DEFAULT 0,
                    total_pnl_money REAL DEFAULT 0,
                    profit_factor REAL DEFAULT 0,
                    max_drawdown_pct REAL DEFAULT 0,
                    sharpe_ratio REAL DEFAULT 0,
                    opt_mode TEXT DEFAULT 'normal',
                    opt_date_from TEXT,
                    opt_date_to TEXT,
                    opt_target_metric TEXT DEFAULT 'profit_factor',
                    is_active INTEGER DEFAULT 1,
                    notes TEXT DEFAULT ''
                );

                CREATE INDEX IF NOT EXISTS idx_opt_strat_lookup
                    ON optimized_strategies(strategy_name, symbol, timeframe, is_active);
            """)

    def save_run(self, result: BacktestResult) -> None:
        """Insert or update a backtest run with its summary metrics."""
        with self._lock, self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO backtest_runs (
                    id, created_at, strategy_name, symbol, timeframe,
                    date_from, date_to, initial_balance, lot_size,
                    sl_pips, tp_pips, total_trades, wins, losses,
                    win_rate, total_pnl_pips, total_pnl_money,
                    profit_factor, max_drawdown, max_drawdown_pct,
                    sharpe_ratio, avg_win_pips, avg_loss_pips,
                    best_trade_pips, worst_trade_pips,
                    avg_trade_duration_mins,
                    max_consecutive_wins, max_consecutive_losses,
                    status, error_msg, progress
                ) VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """, (
                result.run_id,
                datetime.now(timezone.utc).isoformat(),
                result.strategy_name, result.symbol, result.timeframe,
                result.date_from, result.date_to,
                result.initial_balance, result.lot_size,
                result.sl_pips, result.tp_pips,
                result.total_trades, result.wins, result.losses,
                round(result.win_rate, 1),
                round(result.total_pnl_pips, 1),
                round(result.total_pnl_money, 2),
                round(result.profit_factor, 2),
                round(result.max_drawdown, 2),
                round(result.max_drawdown_pct, 1),
                round(result.sharpe_ratio, 2),
                round(result.avg_win_pips, 1),
                round(result.avg_loss_pips, 1),
                round(result.best_trade_pips, 1),
                round(result.worst_trade_pips, 1),
                round(result.avg_trade_duration_mins, 0),
                result.max_consecutive_wins,
                result.max_consecutive_losses,
                result.status, result.error_msg,
                round(result.progress, 2),
            ))

    def save_trades(self, run_id: str, trades: list[BacktestTrade]) -> None:
        """Bulk-insert trades for a run (clears existing first)."""
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM backtest_trades WHERE run_id = ?", (run_id,))
            conn.executemany("""
                INSERT INTO backtest_trades (
                    run_id, entry_time, exit_time, direction,
                    entry_price, exit_price, sl_price, tp_price,
                    lot_size, pnl_pips, pnl_money,
                    exit_reason, signal_confidence, signal_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (run_id, t.entry_time, t.exit_time, t.direction,
                 t.entry_price, t.exit_price, t.sl_price, t.tp_price,
                 t.lot_size, round(t.pnl_pips, 1), round(t.pnl_money, 2),
                 t.exit_reason, t.signal_confidence,
                 (t.signal_reason or "")[:200])
                for t in trades
            ])

    def save_equity_curve(self, run_id: str, curve: list[dict]) -> None:
        """Bulk-insert equity curve points (clears existing first)."""
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM backtest_equity WHERE run_id = ?", (run_id,))
            # Downsample to max 500 points for storage efficiency
            step = max(1, len(curve) // 500)
            sampled = curve[::step]
            if curve and sampled[-1] != curve[-1]:
                sampled.append(curve[-1])
            conn.executemany("""
                INSERT INTO backtest_equity (run_id, bar_time, equity)
                VALUES (?, ?, ?)
            """, [(run_id, pt["time"], round(pt["equity"], 2)) for pt in sampled])

    def get_run(self, run_id: str) -> dict | None:
        """Get a single run summary."""
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM backtest_runs WHERE id = ?", (run_id,)
            ).fetchone()
            return dict(row) if row else None

    def get_trades(self, run_id: str) -> list[dict]:
        """Get all trades for a run."""
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM backtest_trades WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_equity_curve(self, run_id: str) -> list[dict]:
        """Get equity curve for a run."""
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT bar_time as time, equity FROM backtest_equity "
                "WHERE run_id = ? ORDER BY id",
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_history(self, limit: int = 50) -> list[dict]:
        """Get recent backtest runs (summary only)."""
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM backtest_runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def delete_run(self, run_id: str) -> bool:
        """Delete a run and all related data."""
        with self._lock, self._conn() as conn:
            conn.execute("DELETE FROM backtest_equity WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM backtest_trades WHERE run_id = ?", (run_id,))
            cursor = conn.execute(
                "DELETE FROM backtest_runs WHERE id = ?", (run_id,)
            )
            return cursor.rowcount > 0

    # ── Optimized Strategies Repository ───────────────────────────

    def save_optimized_strategy(self, data: dict) -> str:
        """Save an optimized strategy. Deactivates previous active entry
        for the same strategy+symbol+timeframe combo before inserting."""
        import json as _json
        opt_id = "opt-" + str(uuid.uuid4())[:8]
        with self._lock, self._conn() as conn:
            # Deactivate previous active entry for this combo
            conn.execute(
                "UPDATE optimized_strategies SET is_active = 0 "
                "WHERE strategy_name = ? AND symbol = ? AND timeframe = ? AND is_active = 1",
                (data["strategy_name"], data["symbol"], data["timeframe"]),
            )
            conn.execute("""
                INSERT INTO optimized_strategies (
                    id, created_at, strategy_name, symbol, timeframe,
                    params, sl_pips, tp_pips,
                    total_trades, win_rate, total_pnl_pips, total_pnl_money,
                    profit_factor, max_drawdown_pct, sharpe_ratio,
                    opt_mode, opt_date_from, opt_date_to, opt_target_metric,
                    is_active, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?)
            """, (
                opt_id,
                datetime.now(timezone.utc).isoformat(),
                data["strategy_name"],
                data["symbol"],
                data["timeframe"],
                _json.dumps(data.get("params", {})),
                data.get("sl_pips", 0),
                data.get("tp_pips", 0),
                data.get("total_trades", 0),
                round(data.get("win_rate", 0), 1),
                round(data.get("total_pnl_pips", 0), 1),
                round(data.get("total_pnl_money", 0), 2),
                round(data.get("profit_factor", 0), 2),
                round(data.get("max_drawdown_pct", 0), 1),
                round(data.get("sharpe_ratio", 0), 2),
                data.get("opt_mode", "normal"),
                data.get("opt_date_from", ""),
                data.get("opt_date_to", ""),
                data.get("opt_target_metric", "profit_factor"),
                data.get("notes", ""),
            ))
        return opt_id

    def get_optimized_strategy(
        self, strategy_name: str, symbol: str, timeframe: str,
    ) -> dict | None:
        """Get the active optimized strategy for a specific combo."""
        import json as _json
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM optimized_strategies "
                "WHERE strategy_name = ? AND symbol = ? AND timeframe = ? AND is_active = 1 "
                "ORDER BY created_at DESC LIMIT 1",
                (strategy_name, symbol, timeframe),
            ).fetchone()
            if row:
                d = dict(row)
                d["params"] = _json.loads(d.get("params", "{}"))
                return d
        return None

    def get_all_optimized(self) -> list[dict]:
        """Get all saved optimized strategies (active and inactive)."""
        import json as _json
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM optimized_strategies ORDER BY is_active DESC, created_at DESC"
            ).fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["params"] = _json.loads(d.get("params", "{}"))
                results.append(d)
            return results

    def delete_optimized(self, opt_id: str) -> bool:
        """Delete a saved optimized strategy."""
        with self._lock, self._conn() as conn:
            cursor = conn.execute(
                "DELETE FROM optimized_strategies WHERE id = ?", (opt_id,)
            )
            return cursor.rowcount > 0


# =====================================================================
#  Backtest Engine
# =====================================================================

class BacktestEngine:
    """Runs a strategy against historical data and produces results.

    The engine fetches historical candles from MT5, walks through them
    bar-by-bar, calls ``strategy.analyze()`` on a sliding window, and
    simulates trade execution with SL/TP.

    Supports **multiple simultaneous open trades**.
    Uses **next bar's open** for realistic entry (no look-ahead bias).
    """

    # In-flight runs tracked for status polling
    _active_runs: dict[str, BacktestResult] = {}
    _lock = threading.Lock()

    def __init__(self, config: Config | None = None, connector=None):
        self.config = config or Config()
        self.db = BacktestDB()
        # Optional MT5Connector — used to resolve broker symbol suffixes and
        # make the symbol visible in Market Watch before pulling history.
        # Without this, copy_rates_range on a raw/canonical name (e.g. the
        # broker calls it "GBPJPY.r") fails with (-2, 'Invalid params').
        self.connector = connector

    def run_async(
        self,
        strategy_name: str,
        symbol: str,
        timeframe_key: str,
        date_from: str,
        date_to: str,
        initial_balance: float = 10000.0,
        lot_size: float = 0.01,
        sl_pips: float = 0,
        tp_pips: float = 0,
        max_open_trades: int = 5,
        strategy_params: dict | None = None,
    ) -> str:
        """Start a backtest in a background thread. Returns the run_id."""
        run_id = str(uuid.uuid4())[:8]
        result = BacktestResult(
            run_id=run_id,
            strategy_name=strategy_name,
            symbol=symbol,
            timeframe=timeframe_key,
            date_from=date_from,
            date_to=date_to,
            initial_balance=initial_balance,
            lot_size=lot_size,
            sl_pips=sl_pips,
            tp_pips=tp_pips,
            strategy_params=strategy_params or {},
            status="running",
        )

        with self._lock:
            self._active_runs[run_id] = result

        # Save initial run record
        self.db.save_run(result)

        thread = threading.Thread(
            target=self._run_worker,
            args=(result, max_open_trades),
            name=f"Backtest-{run_id}",
            daemon=True,
        )
        thread.start()
        return run_id

    def get_status(self, run_id: str) -> dict | None:
        """Get the status of a running or completed backtest."""
        with self._lock:
            if run_id in self._active_runs:
                r = self._active_runs[run_id]
                return r.to_summary()
        # Check DB
        return self.db.get_run(run_id)

    def get_results(self, run_id: str) -> dict | None:
        """Get full results including trades and equity curve."""
        run = self.db.get_run(run_id)
        if not run:
            return None
        run["trades"] = self.db.get_trades(run_id)
        run["equity_curve"] = self.db.get_equity_curve(run_id)
        return run

    def get_history(self, limit: int = 50) -> list[dict]:
        """Get past backtest run summaries."""
        return self.db.get_history(limit)

    def delete_run(self, run_id: str) -> bool:
        """Delete a backtest run."""
        with self._lock:
            self._active_runs.pop(run_id, None)
        return self.db.delete_run(run_id)

    # ── Worker ────────────────────────────────────────────────────────

    def _run_worker(self, result: BacktestResult, max_open_trades: int) -> None:
        """Execute the backtest in a background thread."""
        try:
            self._execute(result, max_open_trades)
        except Exception as exc:
            logger.error("Backtest %s failed: %s", result.run_id, exc, exc_info=True)
            result.status = "failed"
            result.error_msg = str(exc)[:500]
            self.db.save_run(result)
        finally:
            with self._lock:
                self._active_runs.pop(result.run_id, None)

    def _execute(self, result: BacktestResult, max_open_trades: int) -> None:
        """Core backtest logic."""
        logger.info(
            "🧪 Backtest START: %s on %s %s [%s → %s]",
            result.strategy_name, result.symbol, result.timeframe,
            result.date_from, result.date_to,
        )

        # ── 1. Resolve parameters ────────────────────────────────────
        mt5_tf = TIMEFRAME_MAP.get(result.timeframe)
        if mt5_tf is None:
            raise ValueError(f"Unknown timeframe: {result.timeframe}")

        strategy = get_strategy(result.strategy_name, **result.strategy_params)
        config_sl = result.sl_pips or strategy.sl_pips or self.config.default_sl_pips
        config_tp = result.tp_pips or strategy.tp_pips or self.config.default_tp_pips
        result.sl_pips = config_sl
        result.tp_pips = config_tp

        # Optional time-stop: a strategy may declare ``max_hold_bars`` to force
        # positions closed after N bars (e.g. mean reversion that hasn't
        # reverted). 0 / absent = no time-stop (unchanged behaviour).
        max_hold_bars = int(getattr(strategy, "max_hold_bars", 0) or 0)

        # ── Live-parity gates ─────────────────────────────────────────
        # The live engine blocks entries (a) outside the configured trading
        # hours and (b) during the post-stop-loss cooldown. A backtest that
        # ignores those trades a completely different system (e.g. 24h gold
        # churn with instant re-entry after every stop-out).
        _TF_MINUTES = {
            "M1": 1, "M5": 5, "M15": 15, "M30": 30,
            "H1": 60, "H4": 240, "D1": 1440,
        }
        tf_minutes = _TF_MINUTES.get(result.timeframe, 1)
        cooldown_min = float(getattr(self.config, "sl_cooldown_min", 0) or 0)
        cooldown_bars = int(round(cooldown_min / tf_minutes)) if cooldown_min > 0 else 0
        hour_start = int(getattr(self.config, "trade_hour_start", 0) or 0)
        hour_end = int(getattr(self.config, "trade_hour_end", 24) or 24)
        # NOTE: candle timestamps are broker-SERVER time (often UTC+2/3),
        # while the live gate uses true UTC — treat session results as
        # directional, not exact (calibrate hours from live exports).

        # ── 2. Resolve the broker's real symbol name & enable it ─────
        # Brokers decorate instruments with prefixes/suffixes (GBPJPY →
        # GBPJPY.r, GBPJPYm …) and copy_rates_range fails with
        # (-2, 'Invalid params') if the symbol isn't the exact broker name
        # or isn't selected in Market Watch. Reuse the connector's resolver
        # (same path live trading uses) so backtests accept canonical names.
        broker_symbol = result.symbol
        if self.connector is not None:
            resolved = self.connector.ensure_symbol(result.symbol)
            if resolved is None:
                raise ValueError(
                    f"Symbol '{result.symbol}' is not available on the "
                    f"connected broker (checked common prefixes/suffixes). "
                    f"Open it in MT5 Market Watch or verify the name."
                )
            broker_symbol = resolved
        else:
            # No connector (e.g. standalone script) — at least make the
            # symbol visible so copy_rates_range doesn't fail on selection.
            mt5.symbol_select(result.symbol, True)

        # ── 3. Fetch historical data ─────────────────────────────────
        date_from = datetime.fromisoformat(result.date_from).replace(
            tzinfo=timezone.utc
        )
        date_to = datetime.fromisoformat(result.date_to).replace(
            tzinfo=timezone.utc
        )

        rates = mt5.copy_rates_range(
            broker_symbol, mt5_tf, date_from, date_to,
        )
        if rates is None or len(rates) == 0:
            raise ValueError(
                f"No historical data for {broker_symbol} {result.timeframe} "
                f"from {result.date_from} to {result.date_to}. "
                f"MT5 error: {mt5.last_error()}. On low timeframes like "
                f"{result.timeframe}, MetaTrader may not hold history this far "
                f"back — open the symbol's chart on this timeframe and scroll "
                f"back to force the terminal to download bars, then retry."
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        df = df[["time", "open", "high", "low", "close", "volume"]].copy()

        total_bars = len(df)
        logger.info("Fetched %d bars for backtest.", total_bars)

        # ── 4. Get symbol info for pip calculations ──────────────────
        sym_info = mt5.symbol_info(broker_symbol)
        if sym_info is None:
            raise ValueError(f"Cannot get symbol info for {broker_symbol}")
        point = sym_info.point
        digits = sym_info.digits
        # ONE pip definition for the whole code-base: instruments.pip_size.
        # Backtest and live must agree here or a strategy validated on one
        # trades differently on the other. The previous inline rule read a
        # JPY pip as 10x too large, and an even older variant disagreed
        # with it by 10x on gold, which inflated the spread charge 10x and
        # made every backtested gold trade "lose" ~$6.40 regardless of
        # direction (the 0% win-rate exports).
        pip_size_price = _pip_size(result.symbol, point)
        pip_points = pip_size_price / point if point > 0 else 10.0

        # $ per pip for 1.0 lot. Prefers contract size for an
        # account-currency-quoted symbol, because some feeds report a
        # bogus trade_tick_value while the market is closed.
        pip_value_per_lot = _pip_value_per_lot(
            result.symbol, point,
            contract_size=float(
                getattr(sym_info, "trade_contract_size", 0) or 0
            ),
            tick_value=getattr(sym_info, "trade_tick_value", 0) or 0,
            tick_size=getattr(sym_info, "trade_tick_size", 0) or 0,
        )
        if pip_value_per_lot <= 0:
            pip_value_per_lot = 10.0

        # Spread cost (entry pays the spread). Historical spread isn't in
        # candle data, so we charge the broker's CURRENT spread per trade —
        # but capped at config.max_spread_points: sym_info.spread is a live
        # snapshot, and on a closed market brokers blow it out (weekend gold
        # = 640 points = $6.40!). Live trading never opens a position above
        # the max_spread_points gate (RiskManager), so the simulation must
        # not charge more than that either.
        spread_pips = 0.0
        if point > 0 and getattr(sym_info, "spread", 0):
            max_spread = float(getattr(self.config, "max_spread_points", 30) or 30)
            spread_points = min(float(sym_info.spread), max_spread)
            spread_pips = spread_points / pip_points
        logger.info(
            "Backtest units for %s: point=%s pip=%s $/pip/lot=%.2f "
            "spread=%.1f pips (broker snapshot %s points, capped at "
            "max_spread_points)",
            result.symbol, point, pip_size_price, pip_value_per_lot,
            spread_pips, getattr(sym_info, "spread", 0),
        )

        # ── 5. Walk bar-by-bar ───────────────────────────────────────
        WARMUP = min(200, total_bars // 3)  # indicator warm-up period
        equity = result.initial_balance
        open_trades: list[dict] = []  # active simulated positions
        closed_trades: list[BacktestTrade] = []
        equity_curve: list[dict] = []
        trade_counter = 0
        pending_signal = None  # signal waiting to be executed on next bar
        cooldown_until_index = -1  # no entries until this bar (post-SL cooldown)

        for i in range(WARMUP, total_bars):
            bar = df.iloc[i]
            bar_time = bar["time"].isoformat()

            # ── 5a. Execute pending signal at this bar's open ────────
            if pending_signal is not None:
                sig = pending_signal
                pending_signal = None

                if len(open_trades) < max_open_trades:
                    entry_price = bar["open"]
                    trade_counter += 1

                    if sig.action == SignalAction.BUY:
                        sl_p = round(
                            entry_price - pips_to_price(config_sl, result.symbol, point),
                            digits,
                        )
                        tp_p = round(
                            entry_price + pips_to_price(config_tp, result.symbol, point),
                            digits,
                        )
                        direction = "BUY"
                    else:
                        sl_p = round(
                            entry_price + pips_to_price(config_sl, result.symbol, point),
                            digits,
                        )
                        tp_p = round(
                            entry_price - pips_to_price(config_tp, result.symbol, point),
                            digits,
                        )
                        direction = "SELL"

                    open_trades.append({
                        "id": trade_counter,
                        "direction": direction,
                        "entry_price": entry_price,
                        "entry_time": bar_time,
                        "entry_index": i,
                        "sl_price": sl_p,
                        "tp_price": tp_p,
                        "lot_size": result.lot_size,
                        "confidence": sig.confidence,
                        "reason": sig.reason[:200] if sig.reason else "",
                    })

            # ── 5b. Check SL/TP for all open trades ─────────────────
            still_open = []
            for trade in open_trades:
                hit = self._check_sl_tp(trade, bar, point, result.symbol)
                # Time-stop: if neither SL nor TP was touched and the position
                # has been open at least max_hold_bars, close it at this bar's
                # close (SL/TP take precedence intrabar).
                if not hit and max_hold_bars and (i - trade["entry_index"]) >= max_hold_bars:
                    hit = (bar["close"], "TIME")
                if hit:
                    exit_price, exit_reason = hit
                    pnl_pips = self._calc_pnl_pips(
                        trade["direction"], trade["entry_price"],
                        exit_price, result.symbol, point,
                    ) - spread_pips  # charge the spread
                    pnl_money = pnl_pips * pip_value_per_lot * trade["lot_size"]
                    equity += pnl_money

                    # Post-loss cooldown (mirrors the live engine): a losing
                    # stop-out blocks new entries for a while, killing the
                    # "stopped out, re-enter the same move next bar" churn.
                    if cooldown_bars and exit_reason == "SL" and pnl_money < 0:
                        cooldown_until_index = max(
                            cooldown_until_index, i + cooldown_bars,
                        )

                    closed_trades.append(BacktestTrade(
                        trade_id=trade["id"],
                        entry_time=trade["entry_time"],
                        exit_time=bar_time,
                        direction=trade["direction"],
                        entry_price=trade["entry_price"],
                        exit_price=exit_price,
                        sl_price=trade["sl_price"],
                        tp_price=trade["tp_price"],
                        lot_size=trade["lot_size"],
                        pnl_pips=round(pnl_pips, 1),
                        pnl_money=round(pnl_money, 2),
                        exit_reason=exit_reason,
                        signal_confidence=trade["confidence"],
                        signal_reason=trade["reason"],
                    ))
                else:
                    still_open.append(trade)
            open_trades = still_open

            # ── 5c. Run strategy on current window ───────────────────
            window = df.iloc[max(0, i - 499):i + 1].copy()
            window = window.reset_index(drop=True)

            try:
                signal = strategy.analyze(window, result.symbol)
            except Exception:
                signal = None

            if signal and signal.action in (SignalAction.BUY, SignalAction.SELL):
                # Live-parity gates: trading-hours window + post-SL cooldown.
                bar_hour = int(bar["time"].hour)
                in_hours = (
                    (hour_start <= bar_hour < hour_end)
                    if hour_start <= hour_end
                    else (bar_hour >= hour_start or bar_hour < hour_end)
                )
                if in_hours and i >= cooldown_until_index:
                    pending_signal = signal  # will execute on next bar's open

            # ── 5d. Record equity ────────────────────────────────────
            # Include unrealised P&L from open trades
            unrealised = 0.0
            for trade in open_trades:
                u_pips = self._calc_pnl_pips(
                    trade["direction"], trade["entry_price"],
                    bar["close"], result.symbol, point,
                )
                unrealised += u_pips * pip_value_per_lot * trade["lot_size"]

            equity_curve.append({
                "time": bar_time,
                "equity": round(equity + unrealised, 2),
            })

            # ── 5e. Update progress ──────────────────────────────────
            result.progress = (i - WARMUP) / max(1, total_bars - WARMUP)

        # ── 6. Close any remaining open trades at final bar ──────────
        if open_trades:
            final_bar = df.iloc[-1]
            final_time = final_bar["time"].isoformat()
            for trade in open_trades:
                exit_price = final_bar["close"]
                pnl_pips = self._calc_pnl_pips(
                    trade["direction"], trade["entry_price"],
                    exit_price, result.symbol, point,
                ) - spread_pips  # charge the spread
                pnl_money = pnl_pips * pip_value_per_lot * trade["lot_size"]
                equity += pnl_money

                closed_trades.append(BacktestTrade(
                    trade_id=trade["id"],
                    entry_time=trade["entry_time"],
                    exit_time=final_time,
                    direction=trade["direction"],
                    entry_price=trade["entry_price"],
                    exit_price=exit_price,
                    sl_price=trade["sl_price"],
                    tp_price=trade["tp_price"],
                    lot_size=trade["lot_size"],
                    pnl_pips=round(pnl_pips, 1),
                    pnl_money=round(pnl_money, 2),
                    exit_reason="END",
                    signal_confidence=trade["confidence"],
                    signal_reason=trade["reason"],
                ))

        # ── 7. Compute metrics and save ──────────────────────────────
        result.trades = closed_trades
        result.equity_curve = equity_curve
        result.compute_metrics()
        result.status = "completed"
        result.progress = 1.0

        # Persist to SQLite
        self.db.save_run(result)
        self.db.save_trades(result.run_id, closed_trades)
        self.db.save_equity_curve(result.run_id, equity_curve)

        logger.info(
            "🧪 Backtest DONE: %s | %d trades | Win rate: %.1f%% | "
            "P&L: %.1f pips ($%.2f) | PF: %.2f | Max DD: %.1f%%",
            result.run_id, result.total_trades, result.win_rate,
            result.total_pnl_pips, result.total_pnl_money,
            result.profit_factor, result.max_drawdown_pct,
        )

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _check_sl_tp(
        trade: dict, bar: pd.Series, point: float, symbol: str,
    ) -> tuple[float, str] | None:
        """Check if a bar triggers SL or TP for a trade.

        Delegates to :func:`backtest_core.check_sl_tp` so the in-process
        engine, the sweep and the cloud worker resolve exits identically.
        """
        return check_sl_tp(trade, bar)

    @staticmethod
    def _calc_pnl_pips(
        direction: str,
        entry_price: float,
        exit_price: float,
        symbol: str,
        point: float,
    ) -> float:
        """Calculate P&L in pips (see :func:`backtest_core.calc_pnl_pips`)."""
        return calc_pnl_pips(direction, entry_price, exit_price, symbol, point)


# ======================================================================
#  WALK-FORWARD ANALYSIS (OctoBot-inspired)
# ======================================================================

import random
from dataclasses import dataclass as _dataclass


@_dataclass
class WalkForwardResult:
    """
    Results of walk-forward analysis.

    Attributes
    ----------
    windows : list[dict]
        Per-window results with in_sample_params, metrics, etc.
    robustness_ratio : float
        out_of_sample / in_sample performance.  > 0.5 acceptable,
        > 0.7 good.
    is_robust : bool
        True if robustness_ratio > 0.5.
    """
    windows: list = None
    avg_in_sample_metric: float = 0.0
    avg_out_of_sample_metric: float = 0.0
    robustness_ratio: float = 0.0
    total_out_of_sample_trades: int = 0
    overall_out_of_sample_pnl: float = 0.0
    is_robust: bool = False

    def __post_init__(self):
        if self.windows is None:
            self.windows = []


class WalkForwardAnalyzer:
    """
    Walk-forward analysis for strategy validation.

    Splits data into sliding in-sample / out-of-sample windows,
    optimises on in-sample, validates on out-of-sample.

    Parameters
    ----------
    in_sample_bars : int
        Bars for optimisation window (default 500).
    out_of_sample_bars : int
        Bars for validation window (default 200).
    step_bars : int
        How far to slide each iteration (default 200).
    """

    def __init__(
        self,
        in_sample_bars: int = 500,
        out_of_sample_bars: int = 200,
        step_bars: int = 200,
    ) -> None:
        self.in_sample_bars = in_sample_bars
        self.out_of_sample_bars = out_of_sample_bars
        self.step_bars = step_bars
        self._logger = logging.getLogger("WalkForwardAnalyzer")

    def run(
        self,
        backtest_fn,
        param_combos: list[dict],
        data_df,
        target_metric: str = "profit_factor",
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis.

        Parameters
        ----------
        backtest_fn : callable
            ``fn(params, data_slice) -> dict`` runs a backtest.
        param_combos : list[dict]
            Parameter combinations to try.
        data_df : DataFrame
            Full historical data.
        target_metric : str
            Metric to optimise (default ``'profit_factor'``).

        Returns
        -------
        WalkForwardResult
        """
        total_bars = len(data_df)
        window_size = self.in_sample_bars + self.out_of_sample_bars
        windows: list[dict] = []

        start = 0
        while start + window_size <= total_bars:
            is_start = start
            is_end = start + self.in_sample_bars
            oos_start = is_end
            oos_end = is_end + self.out_of_sample_bars

            in_sample_data = data_df.iloc[is_start:is_end]
            out_of_sample_data = data_df.iloc[oos_start:oos_end]

            # Optimise on in-sample
            best_params = None
            best_metric = float("-inf")

            for params in param_combos:
                try:
                    result = backtest_fn(params, in_sample_data)
                    metric_val = result.get(target_metric, 0.0)
                    if metric_val > best_metric:
                        best_metric = metric_val
                        best_params = params
                except Exception:
                    continue

            if best_params is None:
                start += self.step_bars
                continue

            # Validate on out-of-sample
            try:
                oos_result = backtest_fn(best_params, out_of_sample_data)
                oos_metric = oos_result.get(target_metric, 0.0)
            except Exception:
                oos_metric = 0.0
                oos_result = {}

            windows.append({
                "window_start": is_start,
                "window_end": oos_end,
                "in_sample_params": best_params,
                "in_sample_metric": round(best_metric, 4),
                "out_of_sample_metric": round(oos_metric, 4),
                "out_of_sample_trades": oos_result.get("total_trades", 0),
                "out_of_sample_pnl": oos_result.get("total_pnl_money", 0.0),
            })

            self._logger.info(
                "Window %d: IS=%.4f, OOS=%.4f (params=%s)",
                len(windows), best_metric, oos_metric, best_params,
            )

            start += self.step_bars

        if not windows:
            return WalkForwardResult()

        avg_is = sum(w["in_sample_metric"] for w in windows) / len(windows)
        avg_oos = sum(w["out_of_sample_metric"] for w in windows) / len(windows)
        ratio = avg_oos / avg_is if avg_is > 0 else 0.0
        total_trades = sum(w["out_of_sample_trades"] for w in windows)
        total_pnl = sum(w["out_of_sample_pnl"] for w in windows)

        return WalkForwardResult(
            windows=windows,
            avg_in_sample_metric=round(avg_is, 4),
            avg_out_of_sample_metric=round(avg_oos, 4),
            robustness_ratio=round(ratio, 4),
            total_out_of_sample_trades=total_trades,
            overall_out_of_sample_pnl=round(total_pnl, 2),
            is_robust=ratio > 0.5,
        )


# ======================================================================
#  MONTE CARLO SIMULATION
# ======================================================================

@_dataclass
class MonteCarloResult:
    """
    Monte Carlo simulation results.

    Attributes
    ----------
    probability_of_profit : float
        Percentage of simulations that ended positive.
    """
    num_simulations: int = 0
    confidence_level: float = 0.95
    median_return: float = 0.0
    mean_return: float = 0.0
    worst_case_return: float = 0.0
    best_case_return: float = 0.0
    median_max_drawdown: float = 0.0
    worst_case_max_drawdown: float = 0.0
    probability_of_profit: float = 0.0
    percentile_5_return: float = 0.0
    percentile_95_return: float = 0.0
    equity_curves: list = None

    def __post_init__(self):
        if self.equity_curves is None:
            self.equity_curves = []


class MonteCarloSimulator:
    """
    Monte Carlo simulation by shuffling trade sequences.

    Parameters
    ----------
    num_simulations : int
        Number of randomised equity curves (default 1000).
    confidence_level : float
        Confidence interval (default 0.95).
    """

    def __init__(
        self,
        num_simulations: int = 1000,
        confidence_level: float = 0.95,
    ) -> None:
        self.num_simulations = num_simulations
        self.confidence_level = confidence_level
        self._logger = logging.getLogger("MonteCarloSimulator")

    def run(
        self,
        trades: list[dict],
        initial_balance: float = 10000.0,
    ) -> MonteCarloResult:
        """
        Run Monte Carlo simulation.

        Parameters
        ----------
        trades : list[dict]
            Each dict must have ``'pnl_money'`` key.
        initial_balance : float
            Starting equity.

        Returns
        -------
        MonteCarloResult
        """
        if not trades:
            return MonteCarloResult()

        pnls = [t.get("pnl_money", 0.0) for t in trades]
        final_returns: list[float] = []
        max_drawdowns: list[float] = []
        curves: list[list[float]] = []

        for sim in range(self.num_simulations):
            shuffled = list(pnls)
            random.shuffle(shuffled)

            # Build equity curve
            equity = initial_balance
            peak = equity
            max_dd = 0.0
            curve = [equity]

            for pnl in shuffled:
                equity += pnl
                curve.append(equity)
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak * 100 if peak > 0 else 0
                if dd > max_dd:
                    max_dd = dd

            final_return = ((equity - initial_balance) / initial_balance) * 100
            final_returns.append(final_return)
            max_drawdowns.append(max_dd)

            # Store a subset of curves for visualisation
            if len(curves) < 100:
                curves.append(curve)

        # Compute statistics
        final_returns.sort()
        max_drawdowns.sort()
        n = len(final_returns)

        p5_idx = max(0, int(n * 0.05))
        p95_idx = min(n - 1, int(n * 0.95))
        median_idx = n // 2

        positive_count = sum(1 for r in final_returns if r > 0)

        return MonteCarloResult(
            num_simulations=self.num_simulations,
            confidence_level=self.confidence_level,
            median_return=round(final_returns[median_idx], 2),
            mean_return=round(sum(final_returns) / n, 2),
            worst_case_return=round(final_returns[p5_idx], 2),
            best_case_return=round(final_returns[p95_idx], 2),
            median_max_drawdown=round(max_drawdowns[median_idx], 2),
            worst_case_max_drawdown=round(max_drawdowns[min(n - 1, int(n * 0.95))], 2),
            probability_of_profit=round(positive_count / n * 100, 1),
            percentile_5_return=round(final_returns[p5_idx], 2),
            percentile_95_return=round(final_returns[p95_idx], 2),
            equity_curves=curves,
        )

