"""
news_cache.py - SQLite-backed cache for news articles and AI analysis results.

Provides the ``NewsCache`` class that persists:

* **Articles** – raw news items ingested from various feeds/APIs.
* **Analyses** – structured AI (Claude) sentiment / impact assessments per
  article.
* **Signals** – aggregated BUY / SELL / HOLD signals derived from one or
  more analysed articles.

Thread safety is guaranteed by an internal ``threading.Lock`` that serialises
every database operation.  The SQLite connection is opened with
``check_same_thread=False`` so it can be shared across the bot's worker
threads.

Tables, indexes, and a ``data/`` directory for the database file are created
automatically on first instantiation.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Resolve the project root (mt5_forex_bot/) from this file's location.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"


# =====================================================================
# SQL constants
# =====================================================================

_CREATE_ARTICLES = """
CREATE TABLE IF NOT EXISTS articles (
    id            TEXT PRIMARY KEY,
    headline      TEXT NOT NULL,
    summary       TEXT,
    source        TEXT,
    source_name   TEXT,
    url           TEXT,
    published_at  TEXT,
    symbols       TEXT,          -- JSON array
    category      TEXT,
    fingerprint   TEXT UNIQUE,
    ingested_at   TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

_CREATE_ANALYSES = """
CREATE TABLE IF NOT EXISTS analyses (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    article_id       TEXT    NOT NULL REFERENCES articles(id),
    sentiment        TEXT,   -- BULLISH, BEARISH, NEUTRAL
    confidence       REAL,
    impact_score     REAL,
    urgency          TEXT,
    affected_symbols TEXT,   -- JSON array
    event_type       TEXT,
    reasoning        TEXT,
    suggested_action TEXT,
    time_horizon     TEXT,
    analyzed_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    raw_response     TEXT   -- Full JSON from Claude
);
"""

_CREATE_SIGNALS = """
CREATE TABLE IF NOT EXISTS signals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol          TEXT NOT NULL,
    action          TEXT,   -- BUY, SELL, HOLD
    strength        REAL,
    confidence      REAL,
    source_articles TEXT,   -- JSON array of article IDs
    reasoning       TEXT,
    created_at      TEXT DEFAULT CURRENT_TIMESTAMP,
    expires_at      TEXT
);
"""

_CREATE_USAGE_LOG = """
CREATE TABLE IF NOT EXISTS usage_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    api_calls   INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    logged_at   TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_articles_fingerprint  ON articles  (fingerprint);",
    "CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles  (published_at);",
    "CREATE INDEX IF NOT EXISTS idx_analyses_article_id   ON analyses  (article_id);",
    "CREATE INDEX IF NOT EXISTS idx_signals_symbol        ON signals   (symbol);",
    "CREATE INDEX IF NOT EXISTS idx_signals_created_at    ON signals   (created_at);",
]


# =====================================================================
# NewsCache
# =====================================================================

class NewsCache:
    """SQLite-backed, thread-safe cache for news articles and AI analyses.

    Parameters
    ----------
    db_path:
        Filename (or relative path) for the SQLite database.  It is resolved
        relative to ``<project_root>/data/``.  The directory is created if it
        does not exist.
    retention_days:
        Default number of days to keep articles before ``cleanup()`` removes
        them.
    """

    # ------------------------------------------------------------------ init
    def __init__(self, db_path: str = "news_cache.db", retention_days: int = 30) -> None:
        self._retention_days = retention_days
        self._lock = threading.Lock()

        # Ensure the data/ directory exists.
        _DATA_DIR.mkdir(parents=True, exist_ok=True)

        resolved_path = _DATA_DIR / db_path
        logger.info("Opening news cache database at %s", resolved_path)

        self._conn = sqlite3.connect(
            str(resolved_path),
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row  # dict-like row access
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA foreign_keys=ON;")

        self._create_schema()

    # ------------------------------------------------------------ schema
    def _create_schema(self) -> None:
        """Create tables and indexes if they do not already exist."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(_CREATE_ARTICLES)
            cur.execute(_CREATE_ANALYSES)
            cur.execute(_CREATE_SIGNALS)
            cur.execute(_CREATE_USAGE_LOG)
            for idx_sql in _INDEXES:
                cur.execute(idx_sql)
            self._conn.commit()
            logger.debug("Database schema verified / created.")

    # ================================================================
    # Public API – storage
    # ================================================================

    def store_article(self, article: Any) -> bool:
        """Persist a ``NewsArticle`` object.

        Parameters
        ----------
        article:
            An object exposing at least the attributes: ``id``,
            ``headline``, ``summary``, ``source``, ``source_name``,
            ``url``, ``published_at``, ``symbols``, ``category``,
            ``fingerprint``.

        Returns
        -------
        bool
            ``True`` if the article was inserted; ``False`` if a row with the
            same fingerprint already exists (duplicate suppressed).
        """
        with self._lock:
            try:
                self._conn.execute(
                    """
                    INSERT INTO articles
                        (id, headline, summary, source, source_name,
                         url, published_at, symbols, category, fingerprint)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        getattr(article, "id", None) or str(uuid.uuid4()),
                        article.headline,
                        getattr(article, "summary", None),
                        getattr(article, "source", None),
                        getattr(article, "source_name", None),
                        getattr(article, "url", None),
                        _iso(getattr(article, "published_at", None)),
                        json.dumps(getattr(article, "symbols", []) or []),
                        getattr(article, "category", None),
                        getattr(article, "fingerprint", None),
                    ),
                )
                self._conn.commit()
                logger.debug("Stored article %s: %s", article.id, article.headline)
                return True
            except sqlite3.IntegrityError:
                # Duplicate fingerprint (UNIQUE constraint) → skip silently.
                logger.debug(
                    "Duplicate fingerprint, skipping article: %s",
                    getattr(article, "headline", "?"),
                )
                return False

    def store_analysis(self, article_id: str, analysis: dict) -> int:
        """Store an AI analysis result for a given article.

        Parameters
        ----------
        article_id:
            Foreign key referencing ``articles.id``.
        analysis:
            Dictionary with any of the keys: ``sentiment``, ``confidence``,
            ``impact_score``, ``urgency``, ``affected_symbols``,
            ``event_type``, ``reasoning``, ``suggested_action``,
            ``time_horizon``, ``raw_response``.

        Returns
        -------
        int
            The auto-generated analysis row ID.
        """
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO analyses
                    (article_id, sentiment, confidence, impact_score,
                     urgency, affected_symbols, event_type, reasoning,
                     suggested_action, time_horizon, raw_response)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article_id,
                    analysis.get("sentiment"),
                    analysis.get("confidence"),
                    analysis.get("impact_score"),
                    analysis.get("urgency"),
                    json.dumps(analysis.get("affected_symbols", []) or []),
                    analysis.get("event_type"),
                    analysis.get("reasoning"),
                    analysis.get("suggested_action"),
                    analysis.get("time_horizon"),
                    json.dumps(analysis.get("raw_response"))
                    if analysis.get("raw_response") is not None
                    else None,
                ),
            )
            self._conn.commit()
            analysis_id: int = cur.lastrowid  # type: ignore[assignment]
            logger.debug(
                "Stored analysis %d for article %s", analysis_id, article_id,
            )
            return analysis_id

    def store_signal(
        self,
        symbol: str,
        action: str,
        strength: float,
        confidence: float,
        source_article_ids: list,
        reasoning: str,
        expires_hours: int = 4,
    ) -> int:
        """Create a new trading signal derived from analysed articles.

        Parameters
        ----------
        symbol:
            Forex pair, e.g. ``"EURUSD"``.
        action:
            One of ``BUY``, ``SELL``, ``HOLD``.
        strength:
            Signal strength in [0, 1].
        confidence:
            Model confidence in [0, 1].
        source_article_ids:
            List of ``articles.id`` values that contributed to this signal.
        reasoning:
            Human-readable summary of why the signal was generated.
        expires_hours:
            Number of hours until the signal is considered stale.

        Returns
        -------
        int
            The auto-generated signal row ID.
        """
        expires_at = datetime.now(timezone.utc) + timedelta(hours=expires_hours)
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO signals
                    (symbol, action, strength, confidence,
                     source_articles, reasoning, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    action,
                    strength,
                    confidence,
                    json.dumps(source_article_ids or []),
                    reasoning,
                    expires_at.isoformat(),
                ),
            )
            self._conn.commit()
            signal_id: int = cur.lastrowid  # type: ignore[assignment]
            logger.info(
                "Signal #%d  %s %s  str=%.2f  conf=%.2f  expires=%s",
                signal_id, action, symbol, strength, confidence,
                expires_at.strftime("%H:%M UTC"),
            )
            return signal_id

    # ================================================================
    # Public API – queries
    # ================================================================

    def has_fingerprint(self, fingerprint: str) -> bool:
        """Fast duplicate check by article fingerprint.

        Parameters
        ----------
        fingerprint:
            Content hash / fingerprint string.

        Returns
        -------
        bool
            ``True`` if an article with this fingerprint already exists.
        """
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM articles WHERE fingerprint = ? LIMIT 1",
                (fingerprint,),
            ).fetchone()
            return row is not None

    def get_recent_articles(
        self, limit: int = 50, hours: int = 24,
    ) -> List[Dict[str, Any]]:
        """Return recent articles with their latest analysis joined.

        Parameters
        ----------
        limit:
            Maximum number of articles to return.
        hours:
            Look-back window in hours from *now* (UTC).

        Returns
        -------
        list[dict]
            Each dict contains article columns plus analysis columns
            (prefixed where they would collide).
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    a.id, a.headline, a.summary, a.source, a.source_name,
                    a.url, a.published_at, a.symbols, a.category,
                    a.fingerprint, a.ingested_at,
                    an.sentiment, an.confidence  AS analysis_confidence,
                    an.impact_score, an.urgency,
                    an.affected_symbols, an.event_type,
                    an.reasoning       AS analysis_reasoning,
                    an.suggested_action, an.time_horizon,
                    an.analyzed_at
                FROM articles a
                LEFT JOIN analyses an
                    ON an.id = (
                        SELECT id FROM analyses
                        WHERE article_id = a.id
                        ORDER BY analyzed_at DESC
                        LIMIT 1
                    )
                WHERE a.ingested_at >= ?
                ORDER BY a.ingested_at DESC
                LIMIT ?
                """,
                (cutoff, limit),
            ).fetchall()
            return [_row_to_dict(r) for r in rows]

    def get_recent_signals(
        self, symbol: Optional[str] = None, limit: int = 20,
    ) -> List[Dict[str, Any]]:
        """Return recent signals, optionally filtered by symbol.

        Parameters
        ----------
        symbol:
            If provided, only signals for this pair are returned.
        limit:
            Maximum number of rows.

        Returns
        -------
        list[dict]
        """
        with self._lock:
            if symbol:
                rows = self._conn.execute(
                    """
                    SELECT * FROM signals
                    WHERE symbol = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (symbol, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT * FROM signals
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            return [_row_to_dict(r) for r in rows]

    def get_unanalyzed_articles(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Return articles that have no corresponding analysis row yet.

        Useful for feeding the AI analysis pipeline.

        Parameters
        ----------
        limit:
            Maximum number of articles to return.

        Returns
        -------
        list[dict]
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT a.*
                FROM articles a
                LEFT JOIN analyses an ON an.article_id = a.id
                WHERE an.id IS NULL
                ORDER BY a.ingested_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            return [_row_to_dict(r) for r in rows]

    def get_signal_stats(self) -> Dict[str, Any]:
        """Aggregate statistics across signals and articles.

        Returns
        -------
        dict
            Keys: ``total_articles``, ``analyzed_articles``,
            ``total_signals``, ``signals_by_action`` (dict),
            ``signals_by_symbol`` (dict).
        """
        with self._lock:
            total_articles = self._conn.execute(
                "SELECT COUNT(*) FROM articles",
            ).fetchone()[0]

            analyzed_articles = self._conn.execute(
                """
                SELECT COUNT(DISTINCT article_id) FROM analyses
                """,
            ).fetchone()[0]

            total_signals = self._conn.execute(
                "SELECT COUNT(*) FROM signals",
            ).fetchone()[0]

            action_rows = self._conn.execute(
                "SELECT action, COUNT(*) AS cnt FROM signals GROUP BY action",
            ).fetchall()
            signals_by_action = {r["action"]: r["cnt"] for r in action_rows}

            symbol_rows = self._conn.execute(
                "SELECT symbol, COUNT(*) AS cnt FROM signals GROUP BY symbol",
            ).fetchall()
            signals_by_symbol = {r["symbol"]: r["cnt"] for r in symbol_rows}

        return {
            "total_articles": total_articles,
            "analyzed_articles": analyzed_articles,
            "total_signals": total_signals,
            "signals_by_action": signals_by_action,
            "signals_by_symbol": signals_by_symbol,
        }

    def log_usage(self, api_calls: int, input_tokens: int, output_tokens: int) -> None:
        """Log API usage for persistent cost tracking."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO usage_log (api_calls, input_tokens, output_tokens) VALUES (?, ?, ?)",
                (api_calls, input_tokens, output_tokens),
            )
            self._conn.commit()

    def get_total_usage(self) -> dict:
        """Return cumulative API usage across all sessions."""
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(api_calls),0), COALESCE(SUM(input_tokens),0), COALESCE(SUM(output_tokens),0) FROM usage_log"
            ).fetchone()
            input_t = row[1]
            output_t = row[2]
            cost = input_t * 3.0 / 1_000_000 + output_t * 15.0 / 1_000_000
            return {
                "total_calls": row[0],
                "total_input_tokens": input_t,
                "total_output_tokens": output_t,
                "estimated_cost_usd": round(cost, 6),
            }

    def get_sentiment_timeline(
        self, hours: int = 24, symbol: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Hourly sentiment aggregation suitable for dashboard charts.

        Parameters
        ----------
        hours:
            How many hours to look back.
        symbol:
            Optional filter – only analyses whose ``affected_symbols``
            JSON array contains this symbol are included.

        Returns
        -------
        list[dict]
            One entry per hour with keys ``hour``, ``avg_confidence``,
            ``avg_impact``, ``bullish``, ``bearish``, ``neutral``,
            ``total``.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        with self._lock:
            if symbol:
                rows = self._conn.execute(
                    """
                    SELECT
                        strftime('%%Y-%%m-%%dT%%H:00:00', an.analyzed_at) AS hour,
                        AVG(an.confidence)  AS avg_confidence,
                        AVG(an.impact_score) AS avg_impact,
                        SUM(CASE WHEN an.sentiment = 'BULLISH' THEN 1 ELSE 0 END) AS bullish,
                        SUM(CASE WHEN an.sentiment = 'BEARISH' THEN 1 ELSE 0 END) AS bearish,
                        SUM(CASE WHEN an.sentiment = 'NEUTRAL' THEN 1 ELSE 0 END) AS neutral,
                        COUNT(*) AS total
                    FROM analyses an
                    WHERE an.analyzed_at >= ?
                      AND an.affected_symbols LIKE ?
                    GROUP BY hour
                    ORDER BY hour
                    """,
                    (cutoff, f'%"{symbol}"%'),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    """
                    SELECT
                        strftime('%%Y-%%m-%%dT%%H:00:00', an.analyzed_at) AS hour,
                        AVG(an.confidence)  AS avg_confidence,
                        AVG(an.impact_score) AS avg_impact,
                        SUM(CASE WHEN an.sentiment = 'BULLISH' THEN 1 ELSE 0 END) AS bullish,
                        SUM(CASE WHEN an.sentiment = 'BEARISH' THEN 1 ELSE 0 END) AS bearish,
                        SUM(CASE WHEN an.sentiment = 'NEUTRAL' THEN 1 ELSE 0 END) AS neutral,
                        COUNT(*) AS total
                    FROM analyses an
                    WHERE an.analyzed_at >= ?
                    GROUP BY hour
                    ORDER BY hour
                    """,
                    (cutoff,),
                ).fetchall()
            return [_row_to_dict(r) for r in rows]

    def get_source_health(self) -> List[Dict[str, Any]]:
        """Per-source statistics: article count, last seen, avg rate.

        Returns
        -------
        list[dict]
            Each entry has ``source_name``, ``article_count``,
            ``last_seen``, ``first_seen``, ``avg_articles_per_hour``.
        """
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT
                    source_name,
                    COUNT(*)           AS article_count,
                    MAX(ingested_at)   AS last_seen,
                    MIN(ingested_at)   AS first_seen
                FROM articles
                GROUP BY source_name
                ORDER BY article_count DESC
                """,
            ).fetchall()

        results: List[Dict[str, Any]] = []
        for r in rows:
            entry = _row_to_dict(r)
            # Compute average articles per hour from the source's time span.
            try:
                first = datetime.fromisoformat(entry["first_seen"])
                last = datetime.fromisoformat(entry["last_seen"])
                span_hours = max((last - first).total_seconds() / 3600.0, 1.0)
                entry["avg_articles_per_hour"] = round(
                    entry["article_count"] / span_hours, 2,
                )
            except (TypeError, ValueError):
                entry["avg_articles_per_hour"] = 0.0
            results.append(entry)
        return results

    # ================================================================
    # Maintenance
    # ================================================================

    def cleanup(self, retention_days: Optional[int] = None) -> int:
        """Remove articles (and their analyses) older than *retention_days*.

        Parameters
        ----------
        retention_days:
            Override the default retention window set at construction.

        Returns
        -------
        int
            Number of article rows deleted.
        """
        days = retention_days if retention_days is not None else self._retention_days
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        with self._lock:
            # Delete orphaned analyses first (foreign-key cascade is not
            # guaranteed on all SQLite builds).
            self._conn.execute(
                """
                DELETE FROM analyses
                WHERE article_id IN (
                    SELECT id FROM articles WHERE ingested_at < ?
                )
                """,
                (cutoff,),
            )
            cur = self._conn.execute(
                "DELETE FROM articles WHERE ingested_at < ?",
                (cutoff,),
            )
            # Also prune expired signals while we're at it.
            self._conn.execute(
                "DELETE FROM signals WHERE expires_at < ?",
                (datetime.now(timezone.utc).isoformat(),),
            )
            self._conn.commit()
            deleted: int = cur.rowcount
            if deleted:
                logger.info(
                    "Cleanup: removed %d articles older than %d days.", deleted, days,
                )
            return deleted

    # ================================================================
    # Lifecycle
    # ================================================================

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            try:
                self._conn.close()
                logger.debug("News cache database connection closed.")
            except Exception:
                logger.exception("Error closing news cache database.")

    def __enter__(self) -> "NewsCache":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


# =====================================================================
# Private helpers
# =====================================================================

def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Convert a ``sqlite3.Row`` to a plain dict, deserialising JSON fields."""
    d = dict(row)
    # Deserialise known JSON-array columns when present.
    for key in ("symbols", "affected_symbols", "source_articles"):
        if key in d and isinstance(d[key], str):
            try:
                d[key] = json.loads(d[key])
            except (json.JSONDecodeError, TypeError):
                pass
    # Deserialise raw_response if it's a JSON string.
    if "raw_response" in d and isinstance(d["raw_response"], str):
        try:
            d["raw_response"] = json.loads(d["raw_response"])
        except (json.JSONDecodeError, TypeError):
            pass
    return d


def _iso(dt: Any) -> Optional[str]:
    """Best-effort ISO-8601 serialisation of a datetime-ish value."""
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)
