"""
Donchian Momentum Strategy — CTA Trend Following
=================================================

Time-series momentum à la Moskowitz–Ooi–Pedersen / the classic Turtle
breakout, with a long-term trend filter.  The edge is the most-replicated
anomaly in the CTA literature: instruments that broke to new highs tend to
keep rising as information diffuses slowly and trend-amplifying flows
(carry unwinds, reserve rebalancing) push price further.

Signal logic
------------
* **BUY**  — close breaks *above* the prior N-bar Donchian high **and** price
  is above a rising long-term EMA (trend agreement).
* **SELL** — close breaks *below* the prior N-bar Donchian low **and** price
  is below a falling long-term EMA.
* **HOLD** — no breakout, or breakout disagrees with the long-term trend.

Confidence scales with how far beyond the channel the breakout closed,
normalised by ATR (a breakout that clears the channel by a full ATR carries
more conviction than one that barely pokes through).

Best universe: trending instruments — XAUUSD, USDJPY, GBPJPY — on H1/H4.
Stops/targets stay configurable; enable ``USE_ATR_STOPS`` for proper
volatility-scaled exits and let the trailing engine ride the right tail.
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import BaseStrategy, Signal, SignalAction, TriggerProximity
from .indicators import adx, atr, donchian, ema

logger = logging.getLogger(__name__)


class DonchianMomentumStrategy(BaseStrategy):
    """
    Donchian-channel breakout trend-following strategy.

    Parameters
    ----------
    channel_period : int
        Donchian lookback for breakout detection (default 55 — long enough
        to filter noise on H1/H4, short enough to catch multi-day FX trends).
    trend_period : int
        Long-term EMA used as the directional filter (default 100).
    min_adx : float
        Minimum ADX required to confirm a trending regime (default 20).
        Breakouts in a flat regime are the main source of whipsaw losses.
    atr_period : int
        ATR lookback used to normalise breakout strength (default 14).
    sl_pips, tp_pips : float
        Default stop / target distances (wide — momentum pays in the tail).
    """

    def __init__(
        self,
        channel_period: int = 55,
        trend_period: int = 100,
        min_adx: float = 20.0,
        atr_period: int = 14,
        sl_pips: float = 50,
        tp_pips: float = 150,
    ) -> None:
        super().__init__(sl_pips=sl_pips, tp_pips=tp_pips)
        if channel_period < 5:
            raise ValueError(f"channel_period must be >= 5, got {channel_period}.")
        self.channel_period = channel_period
        self.trend_period = trend_period
        self.min_adx = min_adx
        self.atr_period = atr_period

    @property
    def name(self) -> str:
        return "Donchian Momentum"

    def _min_candles(self) -> int:
        return max(self.channel_period, self.trend_period) + 2

    def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < self._min_candles():
            logger.debug("[%s] %s — insufficient data (%d/%d)",
                         self.name, symbol, len(df), self._min_candles())
            return self._hold(symbol)
        if not {"high", "low", "close"}.issubset(df.columns):
            logger.error("[%s] DataFrame missing OHLC columns.", self.name)
            return self._hold(symbol)

        dc_high, dc_low = donchian(df, self.channel_period)
        # Use the channel formed *before* the current bar to avoid the current
        # bar's own high/low defining the level it must break (lookahead).
        prev_high = dc_high.iloc[-2]
        prev_low = dc_low.iloc[-2]

        trend = ema(df["close"], self.trend_period)
        atr_v = atr(df, self.atr_period).iloc[-1]
        adx_v = adx(df, self.atr_period).iloc[-1]
        close = df["close"].iloc[-1]

        if any(pd.isna(v) for v in (prev_high, prev_low, atr_v, adx_v, trend.iloc[-1])):
            return self._hold(symbol)
        if atr_v <= 0:
            return self._hold(symbol)

        trend_now = trend.iloc[-1]
        trend_rising = trend_now > trend.iloc[-10]
        regime_ok = adx_v >= self.min_adx

        # ---- Bullish breakout ----
        if close > prev_high and close > trend_now and trend_rising and regime_ok:
            confidence = min((close - prev_high) / atr_v, 1.0)
            reason = (
                f"Bullish {self.channel_period}-bar breakout: close={close:.5f} > "
                f"channel={prev_high:.5f} (+{confidence:.2f} ATR), ADX={adx_v:.1f}, trend up"
            )
            logger.info("[%s] %s — %s", self.name, symbol, reason)
            return Signal(SignalAction.BUY, symbol, self.sl_pips, self.tp_pips,
                          confidence, reason)

        # ---- Bearish breakdown ----
        if close < prev_low and close < trend_now and not trend_rising and regime_ok:
            confidence = min((prev_low - close) / atr_v, 1.0)
            reason = (
                f"Bearish {self.channel_period}-bar breakdown: close={close:.5f} < "
                f"channel={prev_low:.5f} (-{confidence:.2f} ATR), ADX={adx_v:.1f}, trend down"
            )
            logger.info("[%s] %s — %s", self.name, symbol, reason)
            return Signal(SignalAction.SELL, symbol, self.sl_pips, self.tp_pips,
                          confidence, reason)

        logger.debug("[%s] %s — no qualifying breakout (ADX=%.1f).",
                     self.name, symbol, adx_v)
        return self._hold(symbol)

    def get_proximity(self, df: pd.DataFrame, symbol: str) -> TriggerProximity:
        if len(df) < self._min_candles():
            return TriggerProximity()

        dc_high, dc_low = donchian(df, self.channel_period)
        prev_high = dc_high.iloc[-2]
        prev_low = dc_low.iloc[-2]
        atr_v = atr(df, self.atr_period).iloc[-1]
        close = df["close"].iloc[-1]

        if any(pd.isna(v) for v in (prev_high, prev_low, atr_v)) or atr_v <= 0:
            return TriggerProximity()

        dist_up = (prev_high - close) / atr_v       # bars of ATR to upper break
        dist_dn = (close - prev_low) / atr_v
        if dist_up <= dist_dn:
            direction, dist, target = "BUY", dist_up, prev_high
        else:
            direction, dist, target = "SELL", dist_dn, prev_low

        proximity = max(0.0, 1.0 - max(dist, 0.0) / 2.0)  # within 2 ATR → warming up
        return TriggerProximity(
            proximity_pct=round(min(proximity, 1.0), 4),
            direction=direction,
            label=f"{self.channel_period}-bar break: {close:.5f}→{target:.5f}",
            detail=f"{dist:.2f} ATR from breakout",
            indicator_values={
                "close": round(close, 6),
                "channel_high": round(prev_high, 6),
                "channel_low": round(prev_low, 6),
                "atr": round(atr_v, 6),
            },
        )
