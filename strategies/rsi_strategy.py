"""
RSI Mean‑Reversion Strategy
============================

Generates signals when RSI exits overbought / oversold territory,
indicating a potential mean‑reversion opportunity.

Signal logic
------------
* **BUY**  — RSI was *below* the oversold threshold and crosses back
  above it (momentum exhaustion → reversal up).
* **SELL** — RSI was *above* the overbought threshold and crosses back
  below it (momentum exhaustion → reversal down).
* **HOLD** — RSI is between thresholds or has not yet reversed.

Confidence is proportional to how far into the extreme zone RSI
travelled before reversing.
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import BaseStrategy, Signal, SignalAction, TriggerProximity

logger = logging.getLogger(__name__)


class RSIStrategy(BaseStrategy):
    """
    RSI‑based mean‑reversion strategy.

    Parameters
    ----------
    period : int
        RSI lookback window (default 14).
    overbought : float
        Upper threshold (default 70).
    oversold : float
        Lower threshold (default 30).
    sl_pips : float
        Stop‑loss distance in pips.
    tp_pips : float
        Take‑profit distance in pips.
    """

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70,
        oversold: float = 30,
        sl_pips: float = 50,
        tp_pips: float = 100,
    ) -> None:
        super().__init__(sl_pips=sl_pips, tp_pips=tp_pips)

        if oversold >= overbought:
            raise ValueError(
                f"oversold ({oversold}) must be less than "
                f"overbought ({overbought})."
            )

        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "RSI Mean Reversion"

    def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        """
        Detect RSI reversals out of overbought / oversold zones.
        """
        min_candles = self.period + 2
        if len(df) < min_candles:
            logger.debug(
                "[%s] %s — insufficient data (%d/%d candles)",
                self.name, symbol, len(df), min_candles,
            )
            return self._hold(symbol)

        try:
            close: pd.Series = df["close"]
        except KeyError:
            logger.error("[%s] DataFrame is missing the 'close' column.", self.name)
            return self._hold(symbol)

        # ---- Calculate RSI manually ----
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=self.period).mean()

        # Guard against division by zero (flat market)
        rs = gain / loss.replace(0, float("nan"))
        rsi = 100.0 - (100.0 / (1.0 + rs))

        curr_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]

        # Skip if RSI could not be computed
        if pd.isna(curr_rsi) or pd.isna(prev_rsi):
            logger.debug("[%s] %s — RSI contains NaN, skipping.", self.name, symbol)
            return self._hold(symbol)

        # ---- Signal detection ----
        # BUY: previous RSI was below oversold, current RSI crosses back above
        if prev_rsi < self.oversold and curr_rsi >= self.oversold:
            # Confidence: how deep below oversold it went (0‑30 → 1.0‑0.0)
            depth = max(self.oversold - prev_rsi, 0)
            confidence = min(depth / self.oversold, 1.0)
            reason = (
                f"RSI reversal from oversold: prev RSI={prev_rsi:.2f} "
                f"crossed above {self.oversold}, curr RSI={curr_rsi:.2f}"
            )
            logger.info("[%s] %s — %s", self.name, symbol, reason)
            return Signal(
                action=SignalAction.BUY,
                symbol=symbol,
                sl_pips=self.sl_pips,
                tp_pips=self.tp_pips,
                confidence=confidence,
                reason=reason,
            )

        # SELL: previous RSI was above overbought, current RSI crosses back below
        if prev_rsi > self.overbought and curr_rsi <= self.overbought:
            # Confidence: how far above overbought it went (70‑100 → 0.0‑1.0)
            depth = max(prev_rsi - self.overbought, 0)
            confidence = min(depth / (100.0 - self.overbought), 1.0)
            reason = (
                f"RSI reversal from overbought: prev RSI={prev_rsi:.2f} "
                f"crossed below {self.overbought}, curr RSI={curr_rsi:.2f}"
            )
            logger.info("[%s] %s — %s", self.name, symbol, reason)
            return Signal(
                action=SignalAction.SELL,
                symbol=symbol,
                sl_pips=self.sl_pips,
                tp_pips=self.tp_pips,
                confidence=confidence,
                reason=reason,
            )

        logger.debug(
            "[%s] %s — no RSI reversal (RSI=%.2f).", self.name, symbol, curr_rsi,
        )
        return self._hold(symbol)

    def get_proximity(self, df: pd.DataFrame, symbol: str) -> TriggerProximity:
        """Measure how close RSI is to an oversold/overbought reversal."""
        min_candles = self.period + 2
        if len(df) < min_candles:
            return TriggerProximity()

        try:
            close = df["close"]
        except KeyError:
            return TriggerProximity()

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=self.period).mean()
        rs = gain / loss.replace(0, float("nan"))
        rsi = 100.0 - (100.0 / (1.0 + rs))

        curr_rsi = rsi.iloc[-1]
        if pd.isna(curr_rsi):
            return TriggerProximity()

        # Determine which threshold is closer
        dist_oversold = abs(curr_rsi - self.oversold)
        dist_overbought = abs(curr_rsi - self.overbought)
        mid = (self.overbought + self.oversold) / 2
        half_range = (self.overbought - self.oversold) / 2

        if curr_rsi <= mid:
            # Approaching oversold → potential BUY
            direction = "BUY"
            dist = dist_oversold
            threshold = self.oversold
            proximity = max(0.0, 1.0 - dist / half_range)
            label = f"RSI: {curr_rsi:.1f} → {threshold:.0f}"
            detail = f"{dist:.1f} points from oversold threshold ({threshold})"
        else:
            # Approaching overbought → potential SELL
            direction = "SELL"
            dist = dist_overbought
            threshold = self.overbought
            proximity = max(0.0, 1.0 - dist / half_range)
            label = f"RSI: {curr_rsi:.1f} → {threshold:.0f}"
            detail = f"{dist:.1f} points from overbought threshold ({threshold})"

        return TriggerProximity(
            proximity_pct=round(min(proximity, 1.0), 4),
            direction=direction,
            label=label,
            detail=detail,
            indicator_values={
                "rsi": round(curr_rsi, 2),
                "oversold": self.oversold,
                "overbought": self.overbought,
                "dist_to_threshold": round(dist, 2),
            },
        )
