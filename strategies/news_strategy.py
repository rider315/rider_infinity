"""
News Intelligence Strategy
===========================

A :class:`~strategies.base.BaseStrategy` subclass that integrates
AI-driven news analysis signals into the standard trading strategy
pipeline.

The strategy delegates signal generation to a
:class:`~news_intelligence.signal_composer.NewsSignalComposer` instance.
When the composer produces a directional signal (BUY / SELL) for the
requested symbol, it is translated into the canonical ``Signal``
dataclass; otherwise a neutral HOLD signal is returned.

Usage
-----
::

    from news_intelligence.signal_composer import NewsSignalComposer
    from strategies.news_strategy import NewsStrategy

    composer = NewsSignalComposer()
    strategy = NewsStrategy(signal_composer=composer, sl_pips=40, tp_pips=80)

    signal = strategy.analyze(df, "EURUSD")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import pandas as pd

from strategies.base import BaseStrategy, Signal, SignalAction, TriggerProximity

if TYPE_CHECKING:
    from news_intelligence.signal_composer import NewsSignalComposer

logger = logging.getLogger(__name__)


class NewsStrategy(BaseStrategy):
    """Strategy that converts news-intelligence signals into trade signals.

    Parameters
    ----------
    signal_composer : NewsSignalComposer or None
        An already-initialised signal composer.  Can also be set later
        via :meth:`set_signal_composer`.
    sl_pips : float
        Default stop-loss distance in pips.
    tp_pips : float
        Default take-profit distance in pips.
    """

    def __init__(
        self,
        signal_composer: Optional["NewsSignalComposer"] = None,
        sl_pips: float = 50,
        tp_pips: float = 100,
    ) -> None:
        super().__init__(sl_pips=sl_pips, tp_pips=tp_pips)
        self._signal_composer = signal_composer
        logger.info(
            "NewsStrategy initialised  sl=%.1f  tp=%.1f  composer=%s",
            sl_pips,
            tp_pips,
            "attached" if signal_composer else "none",
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Human-readable strategy name."""
        return "News Intelligence"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_signal_composer(self, composer: "NewsSignalComposer") -> None:
        """Attach or replace the signal composer after construction.

        Parameters
        ----------
        composer : NewsSignalComposer
            The composer instance to use for signal generation.
        """
        self._signal_composer = composer
        logger.info("Signal composer attached to NewsStrategy")

    def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Analyse news signals for *symbol* and return a trading signal.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV candle data (not directly used by this strategy, but
            required by the ``BaseStrategy`` interface).
        symbol : str
            The instrument symbol, e.g. ``"EURUSD"``.

        Returns
        -------
        Signal
            A BUY or SELL signal when the composer has an actionable
            news signal, or HOLD otherwise.
        """
        if self._signal_composer is None:
            logger.debug("No signal composer set — returning HOLD for %s", symbol)
            return self._hold(symbol)

        try:
            composed = self._signal_composer.get_signal(symbol)
        except Exception:
            logger.exception(
                "Error retrieving composed signal for %s", symbol
            )
            return self._hold(symbol)

        if composed is None:
            logger.debug("No actionable news signal for %s", symbol)
            return self._hold(symbol)

        # Map the composed signal to a Signal dataclass
        action_str = composed.get("action", "").upper()
        if action_str == "BUY":
            action = SignalAction.BUY
        elif action_str == "SELL":
            action = SignalAction.SELL
        else:
            return self._hold(symbol)

        reason = f"[NEWS] {composed.get('reason', 'News-based signal')}"

        signal = Signal(
            action=action,
            symbol=symbol,
            sl_pips=self.sl_pips,
            tp_pips=self.tp_pips,
            confidence=float(composed.get("confidence", 0.0)),
            reason=reason,
        )

        logger.info(
            "NewsStrategy signal  symbol=%s  action=%s  confidence=%.3f  "
            "articles=%d",
            symbol,
            action.value,
            signal.confidence,
            composed.get("article_count", 0),
        )
        return signal

    def get_proximity(self, df: pd.DataFrame, symbol: str) -> TriggerProximity:
        """Use the news composer's latest sentiment as proximity."""
        if self._signal_composer is None:
            return TriggerProximity(label="No news engine")

        try:
            composed = self._signal_composer.get_signal(symbol)
        except Exception:
            return TriggerProximity(label="News error")

        if composed is None:
            return TriggerProximity(label="No news signal")

        confidence = float(composed.get("confidence", 0.0))
        action_str = composed.get("action", "").upper()

        if action_str == "BUY":
            direction = "BUY"
        elif action_str == "SELL":
            direction = "SELL"
        else:
            direction = "NEUTRAL"

        reason = composed.get("reason", "")
        article_count = composed.get("article_count", 0)

        label = f"News: {confidence:.0%} {direction.lower()}"
        detail = f"{article_count} articles · {reason[:80]}"

        return TriggerProximity(
            proximity_pct=round(confidence, 4),
            direction=direction,
            label=label,
            detail=detail,
            indicator_values={
                "confidence": round(confidence, 4),
                "article_count": article_count,
                "action": action_str,
            },
        )
