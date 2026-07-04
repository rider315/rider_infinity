"""
bot_history.py — Persistent bot lifecycle & performance tracking.

Stores a complete audit trail of every bot that has ever existed,
including removed bots, lifecycle events (created, started, stopped,
removed, error, mode-changed), and cumulative trade counts.

Uses SQLite via ``data/bot_history.db``.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils import setup_logger

_DB_PATH = Path(__file__).resolve().parent / "data" / "bot_history.db"


class BotHistoryDB:
    """Thread-safe SQLite store for bot lifecycle history."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = Path(db_path) if db_path else _DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self.logger = setup_logger("BotHistory")
        self._init_schema()

    # ------------------------------------------------------------------
    #  Connection helpers (one connection per thread)
    # ------------------------------------------------------------------
    def _conn(self) -> sqlite3.Connection:
        """Return a per-thread SQLite connection."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA foreign_keys=ON")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        """Create tables if they don't exist."""
        conn = self._conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS bot_records (
                bot_id          INTEGER PRIMARY KEY,
                symbol          TEXT    NOT NULL,
                strategy_name   TEXT    NOT NULL,
                timeframe_key   TEXT    NOT NULL,
                lot_size        REAL    NOT NULL DEFAULT 0.01,
                lot_mode        TEXT    NOT NULL DEFAULT 'fixed',
                dry_run         INTEGER NOT NULL DEFAULT 1,
                strategy_params TEXT    NOT NULL DEFAULT '{}',
                created_at      TEXT    NOT NULL,
                removed_at      TEXT,
                status          TEXT    NOT NULL DEFAULT 'IDLE',
                total_trades    INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS bot_events (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id    INTEGER NOT NULL,
                event     TEXT    NOT NULL,
                detail    TEXT    NOT NULL DEFAULT '',
                timestamp TEXT    NOT NULL,
                FOREIGN KEY (bot_id) REFERENCES bot_records(bot_id)
            );

            CREATE INDEX IF NOT EXISTS idx_events_bot
                ON bot_events(bot_id, timestamp);

            CREATE TABLE IF NOT EXISTS trades (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                bot_id        INTEGER,
                ticket        INTEGER UNIQUE,
                symbol        TEXT    NOT NULL,
                strategy      TEXT    NOT NULL DEFAULT '',
                action        TEXT    NOT NULL DEFAULT '',
                lot           REAL    NOT NULL DEFAULT 0,
                entry_price   REAL    NOT NULL DEFAULT 0,
                sl            REAL    NOT NULL DEFAULT 0,
                tp            REAL    NOT NULL DEFAULT 0,
                open_time     TEXT    NOT NULL,
                close_time    TEXT,
                exit_price    REAL,
                profit        REAL,
                swap          REAL    DEFAULT 0,
                commission    REAL    DEFAULT 0,
                pips          REAL,
                duration_sec  INTEGER,
                exit_reason   TEXT,
                reason        TEXT    DEFAULT '',
                confidence    REAL,
                status        TEXT    NOT NULL DEFAULT 'OPEN',
                logic_version TEXT    DEFAULT ''
            );

            CREATE INDEX IF NOT EXISTS idx_trades_bot
                ON trades(bot_id, open_time);
            CREATE INDEX IF NOT EXISTS idx_trades_ticket
                ON trades(ticket);
        """)
        # Migration: add logic_version to trades tables created before the
        # logic-fingerprint feature (ALTER is a no-op error if it exists).
        cols = [r[1] for r in conn.execute("PRAGMA table_info(trades)")]
        if "logic_version" not in cols:
            conn.execute("ALTER TABLE trades ADD COLUMN logic_version TEXT DEFAULT ''")
        conn.commit()
        self.logger.info("Bot history DB ready at %s", self._db_path)

    # ------------------------------------------------------------------
    #  Record lifecycle events
    # ------------------------------------------------------------------
    def record_bot_created(
        self,
        bot_id: int,
        symbol: str,
        strategy_name: str,
        timeframe_key: str,
        lot_size: float,
        lot_mode: str,
        dry_run: bool,
        strategy_params: dict | None = None,
    ) -> None:
        """Insert a new bot record + CREATED event."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._conn()
        try:
            conn.execute(
                """INSERT OR IGNORE INTO bot_records
                   (bot_id, symbol, strategy_name, timeframe_key,
                    lot_size, lot_mode, dry_run, strategy_params,
                    created_at, removed_at, status, total_trades)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'IDLE', 0)""",
                (
                    bot_id, symbol, strategy_name, timeframe_key,
                    lot_size, lot_mode, int(dry_run),
                    json.dumps(strategy_params or {}),
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO bot_events (bot_id, event, detail, timestamp) VALUES (?, ?, ?, ?)",
                (bot_id, "CREATED", f"{symbol} | {strategy_name} | {timeframe_key}", now),
            )
            conn.commit()
        except Exception as exc:
            self.logger.warning("record_bot_created failed: %s", exc)

    def record_bot_started(self, bot_id: int) -> None:
        """Log a STARTED event and update status."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._conn()
        try:
            conn.execute("UPDATE bot_records SET status = 'RUNNING' WHERE bot_id = ?", (bot_id,))
            conn.execute(
                "INSERT INTO bot_events (bot_id, event, detail, timestamp) VALUES (?, ?, ?, ?)",
                (bot_id, "STARTED", "", now),
            )
            conn.commit()
        except Exception as exc:
            self.logger.warning("record_bot_started failed: %s", exc)

    def record_bot_stopped(self, bot_id: int, trades_taken: int = 0) -> None:
        """Log a STOPPED event, update status and trade count."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE bot_records SET status = 'STOPPED', total_trades = MAX(total_trades, ?) WHERE bot_id = ?",
                (trades_taken, bot_id),
            )
            conn.execute(
                "INSERT INTO bot_events (bot_id, event, detail, timestamp) VALUES (?, ?, ?, ?)",
                (bot_id, "STOPPED", f"Trades taken: {trades_taken}", now),
            )
            conn.commit()
        except Exception as exc:
            self.logger.warning("record_bot_stopped failed: %s", exc)

    def record_bot_removed(self, bot_id: int, trades_taken: int = 0) -> None:
        """Mark bot as removed + log REMOVED event."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE bot_records SET status = 'REMOVED', removed_at = ?, total_trades = MAX(total_trades, ?) WHERE bot_id = ?",
                (now, trades_taken, bot_id),
            )
            conn.execute(
                "INSERT INTO bot_events (bot_id, event, detail, timestamp) VALUES (?, ?, ?, ?)",
                (bot_id, "REMOVED", f"Final trades: {trades_taken}", now),
            )
            conn.commit()
        except Exception as exc:
            self.logger.warning("record_bot_removed failed: %s", exc)

    def record_bot_error(self, bot_id: int, error_msg: str) -> None:
        """Log an ERROR event."""
        now = datetime.now(timezone.utc).isoformat()
        conn = self._conn()
        try:
            conn.execute("UPDATE bot_records SET status = 'ERROR' WHERE bot_id = ?", (bot_id,))
            conn.execute(
                "INSERT INTO bot_events (bot_id, event, detail, timestamp) VALUES (?, ?, ?, ?)",
                (bot_id, "ERROR", error_msg[:500], now),
            )
            conn.commit()
        except Exception as exc:
            self.logger.warning("record_bot_error failed: %s", exc)

    def record_mode_changed(self, bot_id: int, new_dry_run: bool) -> None:
        """Log a MODE_CHANGED event."""
        now = datetime.now(timezone.utc).isoformat()
        mode = "DRY RUN" if new_dry_run else "LIVE"
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE bot_records SET dry_run = ? WHERE bot_id = ?",
                (int(new_dry_run), bot_id),
            )
            conn.execute(
                "INSERT INTO bot_events (bot_id, event, detail, timestamp) VALUES (?, ?, ?, ?)",
                (bot_id, "MODE_CHANGED", f"Switched to {mode}", now),
            )
            conn.commit()
        except Exception as exc:
            self.logger.warning("record_mode_changed failed: %s", exc)

    def increment_trades(self, bot_id: int) -> None:
        """Increment the total_trades counter for a bot."""
        conn = self._conn()
        try:
            conn.execute(
                "UPDATE bot_records SET total_trades = total_trades + 1 WHERE bot_id = ?",
                (bot_id,),
            )
            conn.commit()
        except Exception as exc:
            self.logger.warning("increment_trades failed: %s", exc)

    # ------------------------------------------------------------------
    #  Per-trade performance (live trades)
    # ------------------------------------------------------------------
    def record_trade_open(
        self,
        bot_id: int,
        ticket: int,
        symbol: str,
        strategy: str,
        action: str,
        lot: float,
        entry_price: float,
        sl: float,
        tp: float,
        reason: str = "",
        confidence: float | None = None,
        open_time: str | None = None,
        logic_version: str = "",
    ) -> None:
        """Insert an OPEN trade row keyed by the MT5 position ``ticket``.

        Called when a live order is accepted by the broker.  The matching
        close is recorded later via :meth:`record_trade_close`.
        ``logic_version`` stamps the trade with the code/settings fingerprint
        active at entry (see ``logic_version.py``) so exports can be split
        before/after a strategy change.
        """
        now = open_time or datetime.now(timezone.utc).isoformat()
        conn = self._conn()
        try:
            conn.execute(
                """INSERT OR IGNORE INTO trades
                   (bot_id, ticket, symbol, strategy, action, lot,
                    entry_price, sl, tp, open_time, reason, confidence,
                    status, logic_version)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN', ?)""",
                (
                    bot_id, ticket, symbol, strategy, action, lot,
                    entry_price, sl, tp, now, reason, confidence,
                    logic_version,
                ),
            )
            conn.commit()
        except Exception as exc:
            self.logger.warning("record_trade_open failed: %s", exc)

    def record_trade_close(
        self,
        ticket: int,
        exit_price: float,
        profit: float,
        swap: float = 0.0,
        commission: float = 0.0,
        pips: float | None = None,
        exit_reason: str = "UNKNOWN",
        close_time: str | None = None,
    ) -> bool:
        """Mark a trade CLOSED and store its outcome.

        Returns ``True`` if a matching OPEN row was updated.  Duration is
        computed from the stored ``open_time`` when available.
        """
        now = close_time or datetime.now(timezone.utc).isoformat()
        conn = self._conn()
        try:
            row = conn.execute(
                "SELECT open_time, status FROM trades WHERE ticket = ?", (ticket,)
            ).fetchone()
            if row is None:
                return False  # not a tracked trade (e.g. opened before this feature)

            duration_sec = None
            try:
                opened = datetime.fromisoformat(row["open_time"])
                closed = datetime.fromisoformat(now)
                duration_sec = int((closed - opened).total_seconds())
            except (ValueError, TypeError):
                pass

            conn.execute(
                """UPDATE trades
                   SET status = 'CLOSED', close_time = ?, exit_price = ?,
                       profit = ?, swap = ?, commission = ?, pips = ?,
                       duration_sec = ?, exit_reason = ?
                   WHERE ticket = ?""",
                (
                    now, exit_price, profit, swap, commission, pips,
                    duration_sec, exit_reason, ticket,
                ),
            )
            conn.commit()
            return True
        except Exception as exc:
            self.logger.warning("record_trade_close failed: %s", exc)
            return False

    def import_closed_trade(
        self,
        ticket: int,
        symbol: str,
        action: str,
        lot: float,
        entry_price: float,
        exit_price: float,
        profit: float,
        open_time: str,
        close_time: str,
        swap: float = 0.0,
        commission: float = 0.0,
        pips: float | None = None,
        exit_reason: str = "UNKNOWN",
        bot_id: int | None = None,
        strategy: str = "",
        sl: float = 0.0,
        tp: float = 0.0,
        reason: str = "",
    ) -> bool:
        """Insert an already-CLOSED trade in one shot (used by MT5 backfill).

        Idempotent: an existing ``ticket`` is left untouched so re-running the
        import never duplicates rows.  Returns ``True`` if a new row was added.
        """
        duration_sec = None
        try:
            opened = datetime.fromisoformat(open_time)
            closed = datetime.fromisoformat(close_time)
            duration_sec = int((closed - opened).total_seconds())
        except (ValueError, TypeError):
            pass
        conn = self._conn()
        try:
            cur = conn.execute(
                """INSERT OR IGNORE INTO trades
                   (bot_id, ticket, symbol, strategy, action, lot, entry_price,
                    sl, tp, open_time, close_time, exit_price, profit, swap,
                    commission, pips, duration_sec, exit_reason, reason, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'CLOSED')""",
                (
                    bot_id, ticket, symbol, strategy, action, lot, entry_price,
                    sl, tp, open_time, close_time, exit_price, profit, swap,
                    commission, pips, duration_sec, exit_reason, reason,
                ),
            )
            conn.commit()
            return cur.rowcount > 0
        except Exception as exc:
            self.logger.warning("import_closed_trade failed: %s", exc)
            return False

    def get_trade_by_ticket(self, ticket: int) -> Optional[Dict[str, Any]]:
        """Return a single trade row by its MT5 ticket, or ``None``."""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM trades WHERE ticket = ?", (ticket,)
        ).fetchone()
        return dict(row) if row else None

    def get_open_trade_tickets(self, bot_id: int) -> List[int]:
        """Return tickets of trades still marked OPEN for a bot."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT ticket FROM trades WHERE bot_id = ? AND status = 'OPEN'",
            (bot_id,),
        ).fetchall()
        return [r["ticket"] for r in rows]

    def get_trades(
        self,
        bot_id: int | None = None,
        start: str | None = None,
        end: str | None = None,
        status: str | None = None,
    ) -> List[Dict[str, Any]]:
        """Return trades, optionally filtered by bot, ISO date range, status.

        ``start``/``end`` are matched against ``open_time`` (ISO 8601), so a
        lexical comparison is sufficient.  Results are oldest-first to make
        equity-curve / drawdown computation straightforward.
        """
        clauses: List[str] = []
        params: List[Any] = []
        if bot_id is not None:
            clauses.append("bot_id = ?")
            params.append(bot_id)
        if start:
            clauses.append("open_time >= ?")
            params.append(start)
        if end:
            clauses.append("open_time <= ?")
            params.append(end)
        if status:
            clauses.append("status = ?")
            params.append(status)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        conn = self._conn()
        rows = conn.execute(
            f"SELECT * FROM trades{where} ORDER BY open_time ASC", params
        ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------
    #  Queries
    # ------------------------------------------------------------------
    def get_all_history(self) -> List[Dict[str, Any]]:
        """Return all bot records (active + removed) with event counts."""
        conn = self._conn()
        rows = conn.execute("""
            SELECT r.*,
                   (SELECT COUNT(*) FROM bot_events e WHERE e.bot_id = r.bot_id) AS event_count
            FROM bot_records r
            ORDER BY r.bot_id DESC
        """).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def get_bot_events(self, bot_id: int, limit: int = 100) -> List[Dict[str, Any]]:
        """Return lifecycle events for a specific bot, newest first."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM bot_events WHERE bot_id = ? ORDER BY timestamp DESC LIMIT ?",
            (bot_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_bot_record(self, bot_id: int) -> Optional[Dict[str, Any]]:
        """Return a single bot record by ID."""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM bot_records WHERE bot_id = ?", (bot_id,)
        ).fetchone()
        return self._row_to_dict(row) if row else None

    def get_total_trades(self, bot_id: int) -> int:
        """Get the persisted total_trades for a bot."""
        conn = self._conn()
        row = conn.execute(
            "SELECT total_trades FROM bot_records WHERE bot_id = ?", (bot_id,)
        ).fetchone()
        return row["total_trades"] if row else 0

    def get_removed_bots(self) -> List[Dict[str, Any]]:
        """Return only removed/archived bot records."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM bot_records WHERE removed_at IS NOT NULL ORDER BY removed_at DESC"
        ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        """Convert a sqlite3.Row to a plain dict with parsed JSON fields."""
        d = dict(row)
        # Parse strategy_params back to dict
        if "strategy_params" in d and isinstance(d["strategy_params"], str):
            try:
                d["strategy_params"] = json.loads(d["strategy_params"])
            except (json.JSONDecodeError, TypeError):
                d["strategy_params"] = {}
        # Convert dry_run int to bool
        if "dry_run" in d:
            d["dry_run"] = bool(d["dry_run"])
        return d
