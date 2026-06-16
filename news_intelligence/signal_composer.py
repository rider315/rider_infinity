"""
News Signal Composer
====================

Converts raw AI analysis results into quantitative trading signals.

Each analysis dict is expected to carry at least::

    {
        "sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",
        "confidence": float,       # 0.0 → 1.0
        "impact_score": float,     # 0.0 → 1.0
        "affected_symbols": list[str],
        "reason": str,
    }

The composer aggregates recent analyses per symbol, applies a
time-decay weighting, and emits a net directional signal only
when the weighted score exceeds a configurable threshold.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class NewsSignalComposer:
    """Aggregates AI-generated news analyses into actionable trading signals.

    Parameters
    ----------
    lookback_hours : int
        Maximum age (in hours) of analyses to consider.
    min_confidence : float
        Minimum ``confidence`` an analysis must have to contribute.
    min_impact : float
        Minimum ``impact_score`` an analysis must have to contribute.
    signal_threshold : float
        Absolute net score that must be exceeded before a BUY / SELL
        signal is emitted.
    """

    def __init__(
        self,
        lookback_hours: int = 4,
        min_confidence: float = 0.6,
        min_impact: float = 0.5,
        signal_threshold: float = 0.3,
    ) -> None:
        self._lookback_hours = lookback_hours
        self._min_confidence = min_confidence
        self._min_impact = min_impact
        self._signal_threshold = signal_threshold

        self._analyses: List[Dict[str, Any]] = []
        self._lock = threading.Lock()

        logger.info(
            "NewsSignalComposer initialised  lookback=%dh  min_conf=%.2f  "
            "min_impact=%.2f  threshold=%.2f",
            lookback_hours,
            min_confidence,
            min_impact,
            signal_threshold,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_analysis(self, analysis: Dict[str, Any], article_id: str = "") -> None:
        """Store an analysis result for later signal composition.

        Parameters
        ----------
        analysis : dict
            Raw analysis output from the AI analyser.  Must contain at
            least ``sentiment``, ``confidence``, ``impact_score``,
            ``affected_symbols``, and ``reason``.
        article_id : str, optional
            Optional identifier for the source article.
        """
        entry = dict(analysis)  # shallow copy to avoid mutating caller's dict

        if "analyzed_at" not in entry:
            entry["analyzed_at"] = datetime.now(timezone.utc).isoformat()

        if article_id:
            entry["article_id"] = article_id

        with self._lock:
            self._analyses.append(entry)

        logger.debug(
            "Analysis added  article_id=%s  sentiment=%s  symbols=%s",
            article_id,
            entry.get("sentiment"),
            entry.get("affected_symbols"),
        )

    def get_signal(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Compose an aggregate trading signal for *symbol*.

        Returns
        -------
        dict or None
            A signal dictionary with keys ``action``, ``strength``,
            ``confidence``, ``reason``, ``article_count``,
            ``bullish_score``, ``bearish_score``, and ``net_score``.
            Returns a HOLD signal when analyses exist but net score
            is below threshold.  Returns ``None`` only when no
            analyses exist for the symbol at all.
        """
        with self._lock:
            self._cleanup_old()
            analyses_snapshot = list(self._analyses)

        now = datetime.now(timezone.utc)

        # 1. Collect ALL analyses that mention this symbol within lookback
        #    We track two sets: all matching analyses and those that also
        #    pass confidence/impact thresholds.
        all_for_symbol: List[Dict[str, Any]] = []
        qualified: List[Dict[str, Any]] = []

        for a in analyses_snapshot:
            affected = a.get("affected_symbols", [])
            if symbol not in affected:
                continue

            # Must be within the lookback window
            analyzed_at = self._parse_timestamp(a.get("analyzed_at", ""))
            if analyzed_at is None:
                continue
            age_hours = (now - analyzed_at).total_seconds() / 3600.0
            if age_hours > self._lookback_hours:
                continue

            all_for_symbol.append(a)

            # 2. Check confidence and impact thresholds
            confidence = float(a.get("confidence", 0.0))
            impact_score = float(a.get("impact_score", 0.0))
            if confidence >= self._min_confidence and impact_score >= self._min_impact:
                qualified.append(a)

        # No analyses at all for this symbol → truly nothing to report
        if not all_for_symbol:
            return None

        # 3-5. Calculate weighted scores from qualified analyses
        bullish_score = 0.0
        bearish_score = 0.0
        total_confidence = 0.0
        reasons: List[str] = []

        for a in qualified:
            analyzed_at = self._parse_timestamp(a["analyzed_at"])
            assert analyzed_at is not None  # already validated above
            age_hours = (now - analyzed_at).total_seconds() / 3600.0

            time_decay_factor = max(0.1, 1.0 - (age_hours / self._lookback_hours))

            confidence = float(a.get("confidence", 0.0))
            impact_score = float(a.get("impact_score", 0.0))
            weighted = confidence * impact_score * time_decay_factor

            sentiment = a.get("sentiment", "").upper()
            if sentiment == "BULLISH":
                bullish_score += weighted
            elif sentiment == "BEARISH":
                bearish_score += weighted

            total_confidence += confidence

            reason = a.get("reasoning", "") or a.get("reason", "")
            if reason and reason not in reasons:
                reasons.append(reason)

        # Track the most recent analysis timestamp for display
        latest_at = None
        for a in all_for_symbol:
            ts = a.get("analyzed_at", "")
            if ts and (latest_at is None or ts > latest_at):
                latest_at = ts

        # 6. Net score
        net_score = bullish_score - bearish_score

        # 7. Determine action based on threshold
        if abs(net_score) < self._signal_threshold:
            # Analyses exist but don't produce a directional signal → HOLD
            avg_confidence = total_confidence / len(qualified) if qualified else 0.0
            combined_reason = "; ".join(reasons) if reasons else "News analyzed — no strong directional bias detected"
            signal: Dict[str, Any] = {
                "action": "HOLD",
                "strength": 0.0,
                "confidence": round(avg_confidence, 4),
                "reason": combined_reason,
                "article_count": len(all_for_symbol),
                "qualified_count": len(qualified),
                "bullish_score": round(bullish_score, 4),
                "bearish_score": round(bearish_score, 4),
                "net_score": round(net_score, 4),
                "latest_at": latest_at,
            }
            logger.debug(
                "Signal composed  symbol=%s  action=HOLD  "
                "articles=%d (qualified=%d)  net=%.4f < threshold=%.4f",
                symbol, len(all_for_symbol), len(qualified),
                abs(net_score), self._signal_threshold,
            )
            return signal

        action = "BUY" if net_score > 0 else "SELL"
        strength = min(1.0, abs(net_score))
        avg_confidence = total_confidence / len(qualified) if qualified else 0.0
        combined_reason = "; ".join(reasons) if reasons else "News-based signal"

        signal = {
            "action": action,
            "strength": strength,
            "confidence": round(avg_confidence, 4),
            "reason": combined_reason,
            "article_count": len(all_for_symbol),
            "qualified_count": len(qualified),
            "bullish_score": round(bullish_score, 4),
            "bearish_score": round(bearish_score, 4),
            "net_score": round(net_score, 4),
            "latest_at": latest_at,
        }

        logger.info(
            "Signal composed  symbol=%s  action=%s  strength=%.3f  "
            "confidence=%.3f  articles=%d",
            symbol,
            action,
            strength,
            avg_confidence,
            len(all_for_symbol),
        )
        return signal

    def get_all_signals(self) -> Dict[str, Dict[str, Any]]:
        """Return composed signals for every symbol that has analyses.

        Returns
        -------
        dict[str, dict]
            Mapping of ``{symbol: signal_dict}`` for ALL symbols that
            have analyses — including HOLD signals for symbols below
            the directional threshold.
        """
        # Collect all unique symbols from stored analyses
        with self._lock:
            all_symbols: set[str] = set()
            for a in self._analyses:
                for sym in a.get("affected_symbols", []):
                    all_symbols.add(sym)

        signals: Dict[str, Dict[str, Any]] = {}
        for symbol in sorted(all_symbols):
            signal = self.get_signal(symbol)
            if signal is not None:
                signals[symbol] = signal

        return signals

    def get_actionable_signals(self) -> Dict[str, Dict[str, Any]]:
        """Return only BUY/SELL signals (excluding HOLD).

        Useful for trading logic that only cares about directional signals.

        Returns
        -------
        dict[str, dict]
            Mapping of ``{symbol: signal_dict}`` only for symbols with
            BUY or SELL signals.
        """
        return {
            sym: sig for sym, sig in self.get_all_signals().items()
            if sig.get("action") in ("BUY", "SELL")
        }

    def get_summary(self) -> Dict[str, Any]:
        """Return a concise summary of current composer state.

        Returns
        -------
        dict
            Contains ``total_analyses``, ``active_symbols``,
            ``analyzed_symbols`` (all with analyses including HOLD),
            ``signaling_symbols`` (only BUY/SELL),
            ``active_signals`` (all signals including HOLD),
            and ``has_analyses`` (bool).
        """
        with self._lock:
            total = len(self._analyses)
            symbols: set[str] = set()
            for a in self._analyses:
                for sym in a.get("affected_symbols", []):
                    symbols.add(sym)

        all_signals = self.get_all_signals()
        actionable = {
            sym: sig for sym, sig in all_signals.items()
            if sig.get("action") in ("BUY", "SELL")
        }

        return {
            "total_analyses": total,
            "active_symbols": sorted(symbols),
            "analyzed_symbols": len(all_signals),
            "signaling_symbols": len(actionable),
            "active_signals": all_signals,
            "has_analyses": total > 0,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cleanup_old(self) -> None:
        """Remove analyses older than the lookback window.

        .. note:: Must be called while ``self._lock`` is held.
        """
        now = datetime.now(timezone.utc)
        before = len(self._analyses)

        kept: List[Dict[str, Any]] = []
        for a in self._analyses:
            analyzed_at = self._parse_timestamp(a.get("analyzed_at", ""))
            if analyzed_at is None:
                continue  # drop entries with unparseable timestamps
            age_hours = (now - analyzed_at).total_seconds() / 3600.0
            if age_hours <= self._lookback_hours:
                kept.append(a)

        self._analyses = kept
        removed = before - len(kept)
        if removed:
            logger.debug("Cleaned up %d stale analyses", removed)

    @staticmethod
    def _parse_timestamp(ts: str) -> Optional[datetime]:
        """Parse an ISO-8601 timestamp string into a timezone-aware datetime.

        Returns ``None`` if *ts* is empty or unparseable.
        """
        if not ts:
            return None
        try:
            dt = datetime.fromisoformat(ts)
            # Ensure timezone-aware (assume UTC if naive)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except (ValueError, TypeError):
            logger.warning("Failed to parse timestamp: %s", ts)
            return None

    def __repr__(self) -> str:
        with self._lock:
            count = len(self._analyses)
        return (
            f"<NewsSignalComposer analyses={count} "
            f"lookback={self._lookback_hours}h "
            f"threshold={self._signal_threshold}>"
        )
