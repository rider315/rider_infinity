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
from .indicators import adx, rsi, zscore

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
        return self.z_period + 16  # +ADX(14) warmup headroom

    def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < self._min_candles():
            return self._hold(symbol)
        if "close" not in df.columns:
            logger.error("[%s] DataFrame missing 'close'.", self.name)
            return self._hold(symbol)
        if not self._session_ok(df):
            logger.debug("[%s] %s — outside allowed session.", self.name, symbol)
            return self._hold(symbol)

        z = zscore(df["close"], self.z_period).iloc[-1]
        adx_v = adx(df, 14).iloc[-1]
        rsi_v = rsi(df["close"], self.rsi_period).iloc[-1]

        if any(pd.isna(v) for v in (z, adx_v, rsi_v)):
            return self._hold(symbol)
        if adx_v >= self.max_adx:
            logger.debug("[%s] %s — trending (ADX=%.1f), no fade.", self.name, symbol, adx_v)
            return self._hold(symbol)

        confidence = min(abs(z) / (self.z_threshold + 1.0), 1.0)

        if z <= -self.z_threshold and rsi_v <= self.rsi_buy:
            reason = (f"Oversold fade: z={z:.2f}, RSI({self.rsi_period})={rsi_v:.0f}, "
                      f"ADX={adx_v:.1f} (range)")
            logger.info("[%s] %s — %s", self.name, symbol, reason)
            return Signal(SignalAction.BUY, symbol, self.sl_pips, self.tp_pips,
                          confidence, reason)

        if z >= self.z_threshold and rsi_v >= self.rsi_sell:
            reason = (f"Overbought fade: z={z:.2f}, RSI({self.rsi_period})={rsi_v:.0f}, "
                      f"ADX={adx_v:.1f} (range)")
            logger.info("[%s] %s — %s", self.name, symbol, reason)
            return Signal(SignalAction.SELL, symbol, self.sl_pips, self.tp_pips,
                          confidence, reason)

        return self._hold(symbol)

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
