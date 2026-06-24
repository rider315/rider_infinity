"""
Liquidity-Sweep / Stop-Run Reversal Strategy
==============================================

Stop orders cluster just beyond obvious swing highs/lows.  Price is often
driven to "sweep" that liquidity (large players filling size) and then
*reverses*, because the move was liquidity-motivated rather than
informational.  This strategy fades the exhaustion of such a sweep.

Honest caveat
-------------
Institutions detect sweeps from the *tape and order book*.  This bot only
sees bars, so we use a **sweep-and-reject wick proxy**: a bar that pokes
beyond a recent extreme on an outsized range, then closes back inside,
leaving a long rejection wick.  It is a degraded but real signal — demand a
clean out-of-sample result before trusting it, as the pattern flatters
itself on hand-picked charts.

Signal logic
------------
* **SELL** — bar's high exceeds the prior swing high, closes back below it,
  with a long upper wick on an outsized-range bar, in a non-trending regime.
* **BUY**  — the mirror image at a swept low.
* **HOLD** — otherwise.

Best universe: GBPJPY and majors on M5/M15.  Tight wick-based stops give
good reward:risk when right.
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import BaseStrategy, Signal, SignalAction, TriggerProximity
from .indicators import adx, atr

logger = logging.getLogger(__name__)


class LiquiditySweepStrategy(BaseStrategy):
    """
    Bar-derived stop-run reversal (liquidity-hunting proxy).

    Parameters
    ----------
    swing_lookback : int
        Bars used to define the prior swing high/low that gets swept
        (default 50).  Excludes the current bar.
    wick_ratio : float
        Minimum rejection-wick fraction of the bar's range (default 0.55).
    range_mult : float
        Sweep bar's range must exceed this multiple of ATR — the "size
        moved it" / aggressive-flow proxy (default 1.3).
    max_adx : float
        Only fade when ADX is below this; above it the "sweep" is more
        likely a genuine breakout (default 25).
    atr_period : int
        ATR lookback (default 14).
    sl_pips, tp_pips : float
        Default stop / target (≈2 R; tight stop suits the pattern).
    """

    def __init__(
        self,
        swing_lookback: int = 50,
        wick_ratio: float = 0.55,
        range_mult: float = 1.3,
        max_adx: float = 25.0,
        atr_period: int = 14,
        sl_pips: float = 25,
        tp_pips: float = 50,
    ) -> None:
        super().__init__(sl_pips=sl_pips, tp_pips=tp_pips)
        self.swing_lookback = swing_lookback
        self.wick_ratio = wick_ratio
        self.range_mult = range_mult
        self.max_adx = max_adx
        self.atr_period = atr_period

    @property
    def name(self) -> str:
        return "Liquidity Sweep Reversal"

    def _min_candles(self) -> int:
        return self.swing_lookback + self.atr_period + 2

    def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < self._min_candles():
            return self._hold(symbol)
        if not {"open", "high", "low", "close"}.issubset(df.columns):
            return self._hold(symbol)

        # Prior swing extremes, excluding the current (potential sweep) bar.
        window = df.iloc[-(self.swing_lookback + 1):-1]
        prior_hi = float(window["high"].max())
        prior_lo = float(window["low"].min())

        o, h, l, c = (float(df[col].iloc[-1]) for col in ("open", "high", "low", "close"))
        rng = max(h - l, 1e-9)
        atr_v = atr(df, self.atr_period).iloc[-1]
        adx_v = adx(df, self.atr_period).iloc[-1]
        if pd.isna(atr_v) or pd.isna(adx_v) or atr_v <= 0:
            return self._hold(symbol)
        if adx_v >= self.max_adx:
            logger.debug("[%s] %s — trending (ADX=%.1f), sweep likely real.",
                         self.name, symbol, adx_v)
            return self._hold(symbol)

        upper_wick = (h - max(o, c)) / rng
        lower_wick = (min(o, c) - l) / rng
        big_bar = rng > self.range_mult * atr_v

        swept_high = h > prior_hi and c < prior_hi and upper_wick >= self.wick_ratio and big_bar
        swept_low = l < prior_lo and c > prior_lo and lower_wick >= self.wick_ratio and big_bar

        if swept_high:
            confidence = min(upper_wick, 1.0)
            reason = (f"Sweep + rejection at high {prior_hi:.5f}: upper wick "
                      f"{upper_wick:.2f}, range {rng / atr_v:.1f} ATR -> fade short")
            logger.info("[%s] %s — %s", self.name, symbol, reason)
            return Signal(SignalAction.SELL, symbol, self.sl_pips, self.tp_pips,
                          confidence, reason)

        if swept_low:
            confidence = min(lower_wick, 1.0)
            reason = (f"Sweep + rejection at low {prior_lo:.5f}: lower wick "
                      f"{lower_wick:.2f}, range {rng / atr_v:.1f} ATR -> fade long")
            logger.info("[%s] %s — %s", self.name, symbol, reason)
            return Signal(SignalAction.BUY, symbol, self.sl_pips, self.tp_pips,
                          confidence, reason)

        return self._hold(symbol)

    def get_proximity(self, df: pd.DataFrame, symbol: str) -> TriggerProximity:
        if len(df) < self._min_candles():
            return TriggerProximity()
        window = df.iloc[-(self.swing_lookback + 1):-1]
        prior_hi = float(window["high"].max())
        prior_lo = float(window["low"].min())
        atr_v = atr(df, self.atr_period).iloc[-1]
        close = float(df["close"].iloc[-1])
        if pd.isna(atr_v) or atr_v <= 0:
            return TriggerProximity()

        dist_hi = (prior_hi - close) / atr_v
        dist_lo = (close - prior_lo) / atr_v
        # Proximity to a *level* the market may sweep (then we fade the reverse).
        if abs(dist_hi) <= abs(dist_lo):
            direction, dist, level = "SELL", dist_hi, prior_hi
        else:
            direction, dist, level = "BUY", dist_lo, prior_lo
        proximity = max(0.0, 1.0 - abs(dist) / 1.0)
        return TriggerProximity(
            proximity_pct=round(min(proximity, 1.0), 4),
            direction=direction,
            label=f"Swing level {level:.5f}",
            detail=f"{abs(dist):.2f} ATR from a sweepable extreme",
            indicator_values={"prior_high": round(prior_hi, 6),
                              "prior_low": round(prior_lo, 6),
                              "close": round(close, 6)},
        )
