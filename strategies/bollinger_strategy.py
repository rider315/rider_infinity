"""
Bollinger Bands Strategy — Breakout / Squeeze
===============================================

Generates signals when price reverts back inside the Bollinger Bands
after touching or piercing an outer band, indicating a mean‑reversion
entry point.

Signal logic
------------
* **BUY**  — previous close was *below* the lower band and the current
  close crosses back *above* it (oversold bounce).
* **SELL** — previous close was *above* the upper band and the current
  close crosses back *below* it (overbought rejection).
* **HOLD** — price remains inside the bands or has not reversed.

Confidence is derived from the *bandwidth* (``(upper − lower) / sma``).
A narrower band implies a tighter squeeze — any breakout from that
state carries higher conviction.
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import BaseStrategy, Signal, SignalAction, TriggerProximity

logger = logging.getLogger(__name__)


class BollingerStrategy(BaseStrategy):
    """
    Bollinger Bands mean‑reversion strategy.

    Parameters
    ----------
    period : int
        SMA / standard‑deviation lookback (default 20).
    std_dev : float
        Number of standard deviations for the bands (default 2.0).
    sl_pips : float
        Stop‑loss distance in pips.
    tp_pips : float
        Take‑profit distance in pips.
    """

    def __init__(
        self,
        period: int = 20,
        std_dev: float = 2.0,
        sl_pips: float = 50,
        tp_pips: float = 100,
    ) -> None:
        super().__init__(sl_pips=sl_pips, tp_pips=tp_pips)

        if period < 2:
            raise ValueError(f"period must be >= 2, got {period}.")
        if std_dev <= 0:
            raise ValueError(f"std_dev must be > 0, got {std_dev}.")

        self.period = period
        self.std_dev = std_dev

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "Bollinger Bands"

    def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        """
        Detect price reversals at the Bollinger Band boundaries.
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

        # ---- Calculate Bollinger Bands ----
        sma = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        upper = sma + (self.std_dev * std)
        lower = sma - (self.std_dev * std)

        # Current and previous values
        curr_close = close.iloc[-1]
        prev_close = close.iloc[-2]
        curr_upper = upper.iloc[-1]
        prev_upper = upper.iloc[-2]
        curr_lower = lower.iloc[-1]
        prev_lower = lower.iloc[-2]
        curr_sma = sma.iloc[-1]

        # Guard against NaN (not enough data for rolling window)
        if any(
            pd.isna(v)
            for v in (
                curr_upper, curr_lower, prev_upper, prev_lower, curr_sma,
            )
        ):
            logger.debug(
                "[%s] %s — Bollinger values contain NaN, skipping.", self.name, symbol,
            )
            return self._hold(symbol)

        # ---- Confidence from bandwidth ----
        # Bandwidth = (upper − lower) / sma — smaller → tighter squeeze
        bandwidth = (curr_upper - curr_lower) / curr_sma if curr_sma != 0 else 0
        # Invert so tighter bands → higher confidence, cap at 1.0
        confidence = min(1.0 / (1.0 + bandwidth * 50), 1.0) if bandwidth > 0 else 0.0

        # ---- Signal detection ----
        # BUY: price was below lower band and crossed back above it
        if prev_close < prev_lower and curr_close >= curr_lower:
            reason = (
                f"Bullish Bollinger reversal: price crossed above lower band "
                f"(close={curr_close:.5f}, lower={curr_lower:.5f}, "
                f"bandwidth={bandwidth:.4f})"
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

        # SELL: price was above upper band and crossed back below it
        if prev_close > prev_upper and curr_close <= curr_upper:
            reason = (
                f"Bearish Bollinger reversal: price crossed below upper band "
                f"(close={curr_close:.5f}, upper={curr_upper:.5f}, "
                f"bandwidth={bandwidth:.4f})"
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
            "[%s] %s — price inside bands (close=%.5f, lower=%.5f, upper=%.5f).",
            self.name, symbol, curr_close, curr_lower, curr_upper,
        )
        return self._hold(symbol)

    def get_proximity(self, df: pd.DataFrame, symbol: str) -> TriggerProximity:
        """Measure how close price is to a Bollinger Band boundary."""
        min_candles = self.period + 2
        if len(df) < min_candles:
            return TriggerProximity()

        try:
            close = df["close"]
        except KeyError:
            return TriggerProximity()

        sma = close.rolling(window=self.period).mean()
        std = close.rolling(window=self.period).std()
        upper = sma + (self.std_dev * std)
        lower = sma - (self.std_dev * std)

        curr_close = close.iloc[-1]
        curr_upper = upper.iloc[-1]
        curr_lower = lower.iloc[-1]
        curr_sma = sma.iloc[-1]

        if any(pd.isna(v) for v in (curr_upper, curr_lower, curr_sma)):
            return TriggerProximity()

        band_width = curr_upper - curr_lower
        if band_width == 0:
            return TriggerProximity()

        # Position within bands: 0 = at lower, 1 = at upper
        position = (curr_close - curr_lower) / band_width

        # Near lower band → approaching BUY, near upper → approaching SELL
        if position <= 0.5:
            direction = "BUY"
            dist_to_band = curr_close - curr_lower
            proximity = max(0.0, 1.0 - position * 2)  # 0.0 at middle, 1.0 at lower
            label = f"Price→Lower: {dist_to_band:.5f}"
            detail = (
                f"Close={curr_close:.5f}  Lower={curr_lower:.5f}  "
                f"Distance={dist_to_band:.5f}"
            )
        else:
            direction = "SELL"
            dist_to_band = curr_upper - curr_close
            proximity = max(0.0, (position - 0.5) * 2)  # 0.0 at middle, 1.0 at upper
            label = f"Price→Upper: {dist_to_band:.5f}"
            detail = (
                f"Close={curr_close:.5f}  Upper={curr_upper:.5f}  "
                f"Distance={dist_to_band:.5f}"
            )

        return TriggerProximity(
            proximity_pct=round(min(proximity, 1.0), 4),
            direction=direction,
            label=label,
            detail=detail,
            indicator_values={
                "close": round(curr_close, 6),
                "upper": round(curr_upper, 6),
                "lower": round(curr_lower, 6),
                "sma": round(curr_sma, 6),
                "position": round(position, 4),
            },
        )
