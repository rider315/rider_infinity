"""
Statistical Mean-Reversion Strategy — Session-Conditioned Fade
===============================================================

Fades short-term price displacement on tightly-arbitraged instruments
during *quiet* sessions, when overshoots are driven by transient
liquidity-taking rather than information.  Market makers mean-revert their
inventory and price snaps back toward its short-term mean.  The edge is
being paid to provide liquidity to impatient noise traders; the risk is
that a "noise" move is actually news — hence the trend filter and the
session restriction.

Signal logic
------------
* **BUY**  — price is stretched ≥ ``z_threshold`` std-devs *below* its mean
  **and** a fast RSI is in extreme-oversold territory, in a *non-trending*
  regime (ADX below the gate) during an allowed session.
* **SELL** — the mirror image at the upper extreme.
* **HOLD** — otherwise, or when trending / wrong session.

Confidence scales with the magnitude of the z-score displacement.

Best universe: range-bound majors — EURUSD, EURGBP, USDCHF — on M5/M15.
Note: this strategy has a *fat left tail* (a trend day is its enemy); keep
risk per trade small and let the central daily-drawdown cap protect you.
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import BaseStrategy, Signal, SignalAction, TriggerProximity
from .indicators import adx, atr, ema, rsi, zscore

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Z-score + fast-RSI mean-reversion fade, gated by ADX and session hour.

    Parameters
    ----------
    z_period : int
        Lookback for the rolling mean / std z-score (default 20).
    z_threshold : float
        Displacement (in std-devs) required to fade (default 2.0).
    rsi_period : int
        Fast RSI lookback (default 2 — Connors style; we want *extreme*
        short-term displacement, not a trend reading).
    rsi_buy / rsi_sell : float
        Extreme RSI thresholds (default 5 / 95).
    max_adx : float
        Only fade when ADX is *below* this (no trend) — default 20.
    allowed_hours : set[int] | None
        UTC hours in which fading is permitted.  Defaults to the quiet
        windows (Asian session + London lunch) where reversion dominates.
        Pass ``None`` to disable the session gate.
    use_trend_filter : bool
        When True (default), block fades that lean *against* a clearly
        sloping longer-term trend — i.e. don't buy oversold dips while the
        trend is falling, and don't sell overbought spikes while it's rising.
        This is the fix for the "buying dips in a downtrend" failure mode.
    trend_period : int
        EMA length defining the longer-term trend (default 200).
    trend_slope_lookback : int
        Bars over which the trend EMA's slope is measured (default 20).
    trend_atr_mult : float
        Slope dead-band in ATR units (default 0.5).  A trend counts as
        "clearly sloping" only when the EMA moved more than this × ATR over
        the lookback; flatter than that is treated as a range and both
        directions are allowed.
    max_hold_bars : int
        Time-stop: maximum bars a position may stay open before the engine
        force-closes it (default 16 ≈ 4h on M15).  A fade that hasn't
        reverted quickly has lost its premise.  Read and enforced by the
        backtester / order engine — see ``backtester.py``.
    cooldown_bars : int
        Minimum bars between successive entries on the same symbol (default
        6).  Stops the strategy stacking 3-5 identical fades on consecutive
        bars while the z-score stays beyond the threshold.
    sl_pips, tp_pips : float
        Stop / target.  Reversion runs tighter targets than stops by design
        (high hit-rate, small edge).
    """

    def __init__(
        self,
        z_period: int = 20,
        z_threshold: float = 2.0,
        rsi_period: int = 2,
        rsi_buy: float = 5.0,
        rsi_sell: float = 95.0,
        max_adx: float = 20.0,
        allowed_hours: set[int] | None = frozenset({0, 1, 2, 3, 4, 5, 6, 11, 12}),
        use_trend_filter: bool = True,
        trend_period: int = 200,
        trend_slope_lookback: int = 20,
        trend_atr_mult: float = 0.5,
        max_hold_bars: int = 16,
        cooldown_bars: int = 6,
        sl_pips: float = 40,
        tp_pips: float = 30,
    ) -> None:
        super().__init__(sl_pips=sl_pips, tp_pips=tp_pips)
        if rsi_buy >= rsi_sell:
            raise ValueError("rsi_buy must be < rsi_sell.")
        self.z_period = z_period
        self.z_threshold = z_threshold
        self.rsi_period = rsi_period
        self.rsi_buy = rsi_buy
        self.rsi_sell = rsi_sell
        self.max_adx = max_adx
        self.allowed_hours = set(allowed_hours) if allowed_hours is not None else None
        self.use_trend_filter = use_trend_filter
        self.trend_period = trend_period
        self.trend_slope_lookback = trend_slope_lookback
        self.trend_atr_mult = trend_atr_mult
        self.max_hold_bars = max_hold_bars
        self.cooldown_bars = cooldown_bars
        # Per-symbol timestamp of the last entry we emitted (cooldown state).
        self._last_entry_time: dict[str, pd.Timestamp] = {}

    @property
    def name(self) -> str:
        return "Statistical Mean Reversion"

    def _session_ok(self, df: pd.DataFrame) -> bool:
        if self.allowed_hours is None or "time" not in df.columns:
            return True
        try:
            return int(df["time"].iloc[-1].hour) in self.allowed_hours
        except (AttributeError, TypeError):
            return True  # if time isn't a Timestamp, don't block

    def _min_candles(self) -> int:
        base = self.z_period + 16  # +ADX(14) warmup headroom
        if self.use_trend_filter:
            return max(base, self.trend_period + self.trend_slope_lookback)
        return base

    def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < self._min_candles():
            return self._hold(symbol)
        if "close" not in df.columns:
            logger.error("[%s] DataFrame missing 'close'.", self.name)
            return self._hold(symbol)
        if not self._session_ok(df):
            logger.debug("[%s] %s — outside allowed session.", self.name, symbol)
            return self._hold(symbol)

        close = df["close"]
        z = zscore(close, self.z_period).iloc[-1]
        adx_v = adx(df, 14).iloc[-1]
        rsi_v = rsi(close, self.rsi_period).iloc[-1]

        if any(pd.isna(v) for v in (z, adx_v, rsi_v)):
            return self._hold(symbol)
        if adx_v >= self.max_adx:
            logger.debug("[%s] %s — trending (ADX=%.1f), no fade.", self.name, symbol, adx_v)
            return self._hold(symbol)

        # ---- Raw fade direction ----
        if z <= -self.z_threshold and rsi_v <= self.rsi_buy:
            action = SignalAction.BUY
        elif z >= self.z_threshold and rsi_v >= self.rsi_sell:
            action = SignalAction.SELL
        else:
            return self._hold(symbol)

        # ---- Higher-timeframe trend filter ----
        # Block fades that lean against a clearly sloping trend. A flat trend
        # (|EMA slope| < trend_atr_mult * ATR over the lookback) is treated as
        # a range and allows both directions — exactly where fading works.
        if self.use_trend_filter and len(df) >= self.trend_period + self.trend_slope_lookback:
            trend = ema(close, self.trend_period)
            slope = trend.iloc[-1] - trend.iloc[-1 - self.trend_slope_lookback]
            atr_v = atr(df, 14).iloc[-1]
            if not pd.isna(atr_v) and atr_v > 0:
                band = self.trend_atr_mult * atr_v
                if action == SignalAction.BUY and slope < -band:
                    logger.debug("[%s] %s — BUY blocked: downtrend (slope=%.5f).",
                                 self.name, symbol, slope)
                    return self._hold(symbol)
                if action == SignalAction.SELL and slope > band:
                    logger.debug("[%s] %s — SELL blocked: uptrend (slope=%.5f).",
                                 self.name, symbol, slope)
                    return self._hold(symbol)

        # ---- Entry cooldown (suppress stacked, near-identical fades) ----
        now = df["time"].iloc[-1] if "time" in df.columns else None
        if now is not None and self.cooldown_bars > 0 and len(df) >= 2:
            last = self._last_entry_time.get(symbol)
            try:
                bar_delta = df["time"].iloc[-1] - df["time"].iloc[-2]
                if last is not None and (now - last) < self.cooldown_bars * bar_delta:
                    logger.debug("[%s] %s — cooldown active, skipping.", self.name, symbol)
                    return self._hold(symbol)
            except (TypeError, AttributeError):
                pass  # non-Timestamp index → skip cooldown

        confidence = min(abs(z) / (self.z_threshold + 1.0), 1.0)
        label = "Oversold" if action == SignalAction.BUY else "Overbought"
        reason = (f"{label} fade: z={z:.2f}, RSI({self.rsi_period})={rsi_v:.0f}, "
                  f"ADX={adx_v:.1f} (range)")

        if now is not None:
            self._last_entry_time[symbol] = now

        logger.info("[%s] %s — %s", self.name, symbol, reason)
        return Signal(action, symbol, self.sl_pips, self.tp_pips, confidence, reason)

    def get_proximity(self, df: pd.DataFrame, symbol: str) -> TriggerProximity:
        if len(df) < self._min_candles() or "close" not in df.columns:
            return TriggerProximity()
        z = zscore(df["close"], self.z_period).iloc[-1]
        if pd.isna(z):
            return TriggerProximity()

        direction = "BUY" if z < 0 else "SELL"
        proximity = max(0.0, min(abs(z) / self.z_threshold, 1.0))
        return TriggerProximity(
            proximity_pct=round(proximity, 4),
            direction=direction,
            label=f"z-score: {z:+.2f} → ±{self.z_threshold:.1f}",
            detail=f"{abs(self.z_threshold - abs(z)):.2f} std-devs from fade threshold",
            indicator_values={"zscore": round(float(z), 3), "threshold": self.z_threshold},
        )
