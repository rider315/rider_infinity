"""
MACD Momentum Strategy
=======================

Generates signals on MACD / signal‑line crossovers confirmed by the
histogram direction.

Signal logic
------------
* **BUY**  — MACD line crosses *above* the signal line **and** the
  histogram is positive (momentum shifting bullish).
* **SELL** — MACD line crosses *below* the signal line **and** the
  histogram is negative (momentum shifting bearish).
* **HOLD** — no confirmed crossover on the latest two candles.

Confidence is derived from the absolute magnitude of the histogram
relative to the current price — a larger histogram implies stronger
momentum behind the crossover.
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import BaseStrategy, Signal, SignalAction, TriggerProximity

logger = logging.getLogger(__name__)


class MACDStrategy(BaseStrategy):
    """
    MACD momentum crossover strategy.

    Parameters
    ----------
    fast : int
        Fast EMA period (default 12).
    slow : int
        Slow EMA period (default 26).
    signal : int
        Signal‑line EMA period (default 9).
    sl_pips : float
        Stop‑loss distance in pips.
    tp_pips : float
        Take‑profit distance in pips.
    """

    def __init__(
        self,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9,
        sl_pips: float = 50,
        tp_pips: float = 100,
    ) -> None:
        super().__init__(sl_pips=sl_pips, tp_pips=tp_pips)

        if fast >= slow:
            raise ValueError(
                f"fast ({fast}) must be less than slow ({slow})."
            )

        self.fast = fast
        self.slow = slow
        self.signal = signal

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "MACD Momentum"

    def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        """
        Detect MACD / signal‑line crossovers confirmed by histogram
        direction.
        """
        min_candles = self.slow + self.signal + 2
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

        # ---- Calculate MACD components ----
        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        histogram = macd_line - signal_line

        # Current and previous values
        curr_macd = macd_line.iloc[-1]
        prev_macd = macd_line.iloc[-2]
        curr_signal = signal_line.iloc[-1]
        prev_signal = signal_line.iloc[-2]
        curr_hist = histogram.iloc[-1]

        # Guard against NaN
        if any(pd.isna(v) for v in (curr_macd, prev_macd, curr_signal, prev_signal, curr_hist)):
            logger.debug("[%s] %s — MACD values contain NaN, skipping.", self.name, symbol)
            return self._hold(symbol)

        # ---- Confidence from histogram magnitude ----
        price = close.iloc[-1]
        if price == 0:
            return self._hold(symbol)
        hist_pct = abs(curr_hist) / price
        confidence = min(hist_pct * 1000, 1.0)  # scale up, cap at 1.0

        # ---- Crossover detection ----
        # BUY: MACD crosses above signal line AND histogram is positive
        if prev_macd <= prev_signal and curr_macd > curr_signal and curr_hist > 0:
            reason = (
                f"Bullish MACD crossover: MACD={curr_macd:.6f} crossed above "
                f"signal={curr_signal:.6f}, histogram={curr_hist:.6f}"
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

        # SELL: MACD crosses below signal line AND histogram is negative
        if prev_macd >= prev_signal and curr_macd < curr_signal and curr_hist < 0:
            reason = (
                f"Bearish MACD crossover: MACD={curr_macd:.6f} crossed below "
                f"signal={curr_signal:.6f}, histogram={curr_hist:.6f}"
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
            "[%s] %s — no MACD crossover (MACD=%.6f, signal=%.6f, hist=%.6f).",
            self.name, symbol, curr_macd, curr_signal, curr_hist,
        )
        return self._hold(symbol)

    def get_proximity(self, df: pd.DataFrame, symbol: str) -> TriggerProximity:
        """Measure how close the MACD line is to crossing the signal line."""
        min_candles = self.slow + self.signal + 2
        if len(df) < min_candles:
            return TriggerProximity()

        try:
            close = df["close"]
        except KeyError:
            return TriggerProximity()

        ema_fast = close.ewm(span=self.fast, adjust=False).mean()
        ema_slow = close.ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()

        curr_macd = macd_line.iloc[-1]
        curr_signal = signal_line.iloc[-1]
        price = close.iloc[-1]

        if any(pd.isna(v) for v in (curr_macd, curr_signal)) or price == 0:
            return TriggerProximity()

        gap = abs(curr_macd - curr_signal)

        # Normalise using recent max gap
        lookback = min(50, len(macd_line))
        recent_gaps = abs(macd_line.iloc[-lookback:] - signal_line.iloc[-lookback:])
        max_gap = recent_gaps.max()

        if max_gap == 0:
            proximity = 0.0
        else:
            proximity = 1.0 - min(gap / max_gap, 1.0)

        # Direction: MACD below signal → approaching bullish cross
        direction = "BUY" if curr_macd < curr_signal else "SELL"

        label = f"MACD Gap: {gap:.6f}"
        detail = (
            f"MACD={curr_macd:.6f}  Signal={curr_signal:.6f}  "
            f"Gap={gap/price*100:.4f}%"
        )

        return TriggerProximity(
            proximity_pct=round(proximity, 4),
            direction=direction,
            label=label,
            detail=detail,
            indicator_values={
                "macd": round(curr_macd, 6),
                "signal_line": round(curr_signal, 6),
                "gap": round(gap, 6),
                "gap_pct": round(gap / price * 100, 4),
            },
        )
