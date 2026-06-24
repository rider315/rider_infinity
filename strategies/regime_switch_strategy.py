"""
Regime-Switching Strategy — Self-Adapting Trend / Range
========================================================

A single instrument oscillates between *trending* and *mean-reverting*
regimes.  Pure trend systems bleed in chop; pure reversion systems blow up
in trends.  This strategy detects the regime and applies the matching
logic, standing aside in the ambiguous transition zone.  It is fully
self-contained: it carries its own regime detector, trend rule, and
reversion rule, and depends on no other strategy.

Regime detection
----------------
* **Trend**  — ADX above ``trend_adx`` *and* variance ratio > 1 (returns
  positively autocorrelated).  Trade pullback continuation.
* **Range**  — ADX below ``range_adx`` *and* variance ratio < 1 (returns
  mean-reverting).  Fade band extremes.
* **Transition** — anything in between → HOLD (the whipsaw zone).

Best universe: EURUSD, XAUUSD on M15/H1.  The transition buffer is what
keeps it from flip-flopping at regime boundaries.
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import BaseStrategy, Signal, SignalAction, TriggerProximity
from .indicators import adx, ema, variance_ratio, zscore

logger = logging.getLogger(__name__)


class RegimeSwitchStrategy(BaseStrategy):
    """
    ADX + variance-ratio regime classifier branching to trend or reversion.

    Parameters
    ----------
    fast_ema, slow_ema : int
        Trend-leg moving averages (default 20 / 50).
    trend_adx : float
        ADX above which a trending regime is declared (default 25).
    range_adx : float
        ADX below which a ranging regime is declared (default 18).
    vr_lag : int
        Variance-ratio lag (default 5).
    z_period : int
        Lookback for the reversion-leg z-score (default 20).
    z_threshold : float
        Fade threshold in std-devs for the range leg (default 2.0).
    sl_pips, tp_pips : float
        Default stop / target (the central engine may override per regime
        via ATR stops).
    """

    def __init__(
        self,
        fast_ema: int = 20,
        slow_ema: int = 50,
        trend_adx: float = 25.0,
        range_adx: float = 18.0,
        vr_lag: int = 5,
        z_period: int = 20,
        z_threshold: float = 2.0,
        sl_pips: float = 45,
        tp_pips: float = 110,
    ) -> None:
        super().__init__(sl_pips=sl_pips, tp_pips=tp_pips)
        if fast_ema >= slow_ema:
            raise ValueError("fast_ema must be < slow_ema.")
        if range_adx >= trend_adx:
            raise ValueError("range_adx must be < trend_adx (need a transition buffer).")
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.trend_adx = trend_adx
        self.range_adx = range_adx
        self.vr_lag = vr_lag
        self.z_period = z_period
        self.z_threshold = z_threshold

    @property
    def name(self) -> str:
        return "Regime Switch"

    def _min_candles(self) -> int:
        return max(self.slow_ema, self.z_period, self.vr_lag * 4) + 16

    def _classify(self, df: pd.DataFrame) -> tuple[str, float, float]:
        adx_v = adx(df, 14).iloc[-1]
        vr = variance_ratio(df["close"], self.vr_lag)
        if pd.isna(adx_v):
            return "TRANSITION", float("nan"), vr
        if adx_v >= self.trend_adx and vr > 1.0:
            return "TREND", adx_v, vr
        if adx_v <= self.range_adx and vr < 1.0:
            return "RANGE", adx_v, vr
        return "TRANSITION", adx_v, vr

    def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < self._min_candles() or "close" not in df.columns:
            return self._hold(symbol)

        regime, adx_v, vr = self._classify(df)
        close = float(df["close"].iloc[-1])

        if regime == "TREND":
            fast = ema(df["close"], self.fast_ema)
            slow = ema(df["close"], self.slow_ema)
            f_now, f_prev = fast.iloc[-1], fast.iloc[-2]
            s_now = slow.iloc[-1]
            prev_close = float(df["close"].iloc[-2])
            if any(pd.isna(v) for v in (f_now, f_prev, s_now)):
                return self._hold(symbol)

            # Uptrend: pullback to fast EMA then reclaim (continuation long).
            if f_now > s_now and prev_close < f_prev and close > f_now:
                conf = min((adx_v - self.trend_adx) / 25.0 + 0.5, 1.0)
                reason = (f"TREND-UP pullback reclaim: EMA{self.fast_ema}>EMA{self.slow_ema}, "
                          f"ADX={adx_v:.1f}, VR={vr:.2f}")
                logger.info("[%s] %s — %s", self.name, symbol, reason)
                return Signal(SignalAction.BUY, symbol, self.sl_pips, self.tp_pips,
                              conf, reason)
            if f_now < s_now and prev_close > f_prev and close < f_now:
                conf = min((adx_v - self.trend_adx) / 25.0 + 0.5, 1.0)
                reason = (f"TREND-DOWN pullback reclaim: EMA{self.fast_ema}<EMA{self.slow_ema}, "
                          f"ADX={adx_v:.1f}, VR={vr:.2f}")
                logger.info("[%s] %s — %s", self.name, symbol, reason)
                return Signal(SignalAction.SELL, symbol, self.sl_pips, self.tp_pips,
                              conf, reason)
            return self._hold(symbol)

        if regime == "RANGE":
            z = zscore(df["close"], self.z_period).iloc[-1]
            if pd.isna(z):
                return self._hold(symbol)
            conf = min(abs(z) / (self.z_threshold + 1.0), 1.0)
            if z <= -self.z_threshold:
                reason = f"RANGE fade long: z={z:.2f}, ADX={adx_v:.1f}, VR={vr:.2f}"
                logger.info("[%s] %s — %s", self.name, symbol, reason)
                return Signal(SignalAction.BUY, symbol, self.sl_pips, self.tp_pips,
                              conf, reason)
            if z >= self.z_threshold:
                reason = f"RANGE fade short: z={z:.2f}, ADX={adx_v:.1f}, VR={vr:.2f}"
                logger.info("[%s] %s — %s", self.name, symbol, reason)
                return Signal(SignalAction.SELL, symbol, self.sl_pips, self.tp_pips,
                              conf, reason)
            return self._hold(symbol)

        logger.debug("[%s] %s — TRANSITION (ADX=%.1f, VR=%.2f), standing aside.",
                     self.name, symbol, adx_v if adx_v == adx_v else -1, vr)
        return self._hold(symbol)

    def get_proximity(self, df: pd.DataFrame, symbol: str) -> TriggerProximity:
        if len(df) < self._min_candles() or "close" not in df.columns:
            return TriggerProximity()
        regime, adx_v, vr = self._classify(df)
        if regime == "TRANSITION":
            return TriggerProximity(
                proximity_pct=0.0, direction="NEUTRAL",
                label=f"Regime: TRANSITION (ADX={adx_v if adx_v == adx_v else float('nan'):.1f})",
                detail="Awaiting a clear trend or range regime",
                indicator_values={"regime": regime, "adx": None if adx_v != adx_v else round(adx_v, 1),
                                  "vr": round(vr, 3)},
            )
        if regime == "RANGE":
            z = zscore(df["close"], self.z_period).iloc[-1]
            if pd.isna(z):
                return TriggerProximity()
            direction = "BUY" if z < 0 else "SELL"
            proximity = max(0.0, min(abs(z) / self.z_threshold, 1.0))
            return TriggerProximity(
                proximity_pct=round(proximity, 4), direction=direction,
                label=f"RANGE z={z:+.2f}", detail=f"ADX={adx_v:.1f}, VR={vr:.2f}",
                indicator_values={"regime": regime, "zscore": round(float(z), 3),
                                  "adx": round(adx_v, 1), "vr": round(vr, 3)},
            )
        # TREND: proximity to fast EMA (pullback entry zone)
        fast = ema(df["close"], self.fast_ema)
        slow = ema(df["close"], self.slow_ema)
        close = float(df["close"].iloc[-1])
        direction = "BUY" if fast.iloc[-1] > slow.iloc[-1] else "SELL"
        gap = abs(close - fast.iloc[-1]) / close if close else 1.0
        proximity = max(0.0, 1.0 - gap * 200)  # within ~0.5% of fast EMA = ready
        return TriggerProximity(
            proximity_pct=round(min(proximity, 1.0), 4), direction=direction,
            label=f"TREND pullback to EMA{self.fast_ema}",
            detail=f"ADX={adx_v:.1f}, VR={vr:.2f}",
            indicator_values={"regime": regime, "adx": round(adx_v, 1), "vr": round(vr, 3),
                              "fast_ema": round(float(fast.iloc[-1]), 6)},
        )
