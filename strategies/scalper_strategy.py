"""
Micro-Trend Scalper Strategy
============================

A fast, high-frequency strategy built to *catch the start of a small trend*
("ek chhota sa trend pakdo") and get in early, so it can be paired with the
per-bot **Quick Exit** feature that banks a few pips and closes immediately.

It is deliberately aggressive on entries but uses a multi-factor confluence so
it doesn't fire inside dead, choppy ranges (where scalping just bleeds the
spread).  An entry needs ALL of:

1. **Trend bias** — fast EMA above/below slow EMA (direction of the micro-trend).
2. **Trend is accelerating** — the fast EMA is sloping the right way
   (``ema_fast`` rising for longs, falling for shorts).
3. **Real momentum, not chop** — ADX ≥ ``adx_min`` (Wilder).  Below this the
   market is ranging and breakouts fail.
4. **Momentum push** — short-period rate-of-change agrees with the direction.
5. **Micro-breakout trigger** — the latest close breaks the high/low of the
   previous ``breakout_lookback`` candles, i.e. a fresh thrust is underway.

Confidence scales with how strong the trend (ADX) and the breakout thrust are.

Notes
-----
* All math runs on **closed** candles (the live loop drops the forming bar),
  so signals don't flicker tick-by-tick.
* SL/TP defaults are tight — this strategy is meant to be run with Quick Exit
  ON, which is the real exit.  The central risk engine still clamps/sizes them.
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import BaseStrategy, Signal, SignalAction, TriggerProximity
from .indicators import adx, atr, ema

logger = logging.getLogger(__name__)


class ScalperStrategy(BaseStrategy):
    """Fast micro-trend breakout scalper.

    Parameters
    ----------
    ema_fast : int
        Fast EMA span (trend direction). Default 5.
    ema_slow : int
        Slow EMA span (trend filter). Default 20.
    adx_period : int
        ADX lookback. Default 14.
    adx_min : float
        Minimum ADX to allow an entry (chop filter). Default 25. Raised from
        18 after the live export showed a 45.8% win-rate — most losers were
        low-ADX chop breakouts that immediately reversed. A stricter gate
        trades less but only when a real trend thrust is underway.
    breakout_lookback : int
        How many prior candles define the micro-breakout level. Default 8.
    roc_period : int
        Rate-of-change lookback for the momentum push. Default 3.
    sl_pips : float
        Stop-loss distance in pips (tight). Default 8.
    tp_pips : float
        Take-profit distance in pips (backstop; Quick Exit is the real exit).
        Default 12.
    """

    def __init__(
        self,
        ema_fast: int = 5,
        ema_slow: int = 20,
        adx_period: int = 14,
        adx_min: float = 25.0,
        breakout_lookback: int = 8,
        roc_period: int = 3,
        sl_pips: float = 8,
        tp_pips: float = 12,
    ) -> None:
        super().__init__(sl_pips=sl_pips, tp_pips=tp_pips)

        if ema_fast >= ema_slow:
            raise ValueError(
                f"ema_fast ({ema_fast}) must be less than ema_slow ({ema_slow})."
            )
        if breakout_lookback < 2:
            raise ValueError("breakout_lookback must be >= 2.")

        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.adx_period = adx_period
        self.adx_min = adx_min
        self.breakout_lookback = breakout_lookback
        self.roc_period = roc_period

    # ------------------------------------------------------------------
    @property
    def name(self) -> str:
        return "Micro-Trend Scalper"

    # ------------------------------------------------------------------
    def _min_candles(self) -> int:
        # Need enough history for the slow EMA + ADX smoothing to settle.
        return max(self.ema_slow, self.adx_period) + self.breakout_lookback + 5

    def _compute(self, df: pd.DataFrame):
        """Return the indicator snapshot used by both analyze & proximity."""
        close = df["close"]
        ef = ema(close, self.ema_fast)
        es = ema(close, self.ema_slow)
        adx_series = adx(df, self.adx_period)
        atr_series = atr(df, self.adx_period)

        # Breakout levels = extreme of the *previous* N closed candles
        # (exclude the current candle so "close breaks the level" is a real
        # new high/low, not the candle comparing against itself).
        prior = df.iloc[:-1]
        hi = prior["high"].rolling(self.breakout_lookback).max().iloc[-1]
        lo = prior["low"].rolling(self.breakout_lookback).min().iloc[-1]

        return {
            "close": close.iloc[-1],
            "ema_fast": ef.iloc[-1],
            "ema_fast_prev": ef.iloc[-2],
            "ema_slow": es.iloc[-1],
            "adx": adx_series.iloc[-1],
            "atr": atr_series.iloc[-1],
            "breakout_high": hi,
            "breakout_low": lo,
            "roc": close.iloc[-1] - close.iloc[-1 - self.roc_period],
        }

    # ------------------------------------------------------------------
    def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < self._min_candles():
            return self._hold(symbol)
        if "close" not in df or "high" not in df or "low" not in df:
            logger.error("[%s] DataFrame missing OHLC columns.", self.name)
            return self._hold(symbol)

        s = self._compute(df)

        # Bail out if any indicator is NaN (warm-up period).
        if any(pd.isna(v) for v in s.values()):
            return self._hold(symbol)

        adx_ok = s["adx"] >= self.adx_min
        uptrend = s["ema_fast"] > s["ema_slow"]
        downtrend = s["ema_fast"] < s["ema_slow"]
        ema_rising = s["ema_fast"] > s["ema_fast_prev"]
        ema_falling = s["ema_fast"] < s["ema_fast_prev"]

        # ---- BUY: uptrend + accelerating + momentum + micro-breakout up ----
        if (
            adx_ok
            and uptrend
            and ema_rising
            and s["roc"] > 0
            and s["close"] >= s["breakout_high"]
        ):
            conf = self._confidence(s)
            reason = (
                f"Micro-trend long: EMA{self.ema_fast}>{self.ema_slow}, "
                f"ADX={s['adx']:.1f}, close {s['close']:.5f} broke "
                f"{self.breakout_lookback}-bar high {s['breakout_high']:.5f}"
            )
            logger.info("[%s] %s — %s", self.name, symbol, reason)
            return Signal(
                action=SignalAction.BUY, symbol=symbol,
                sl_pips=self.sl_pips, tp_pips=self.tp_pips,
                confidence=conf, reason=reason,
            )

        # ---- SELL: downtrend + accelerating + momentum + micro-breakout dn --
        if (
            adx_ok
            and downtrend
            and ema_falling
            and s["roc"] < 0
            and s["close"] <= s["breakout_low"]
        ):
            conf = self._confidence(s)
            reason = (
                f"Micro-trend short: EMA{self.ema_fast}<{self.ema_slow}, "
                f"ADX={s['adx']:.1f}, close {s['close']:.5f} broke "
                f"{self.breakout_lookback}-bar low {s['breakout_low']:.5f}"
            )
            logger.info("[%s] %s — %s", self.name, symbol, reason)
            return Signal(
                action=SignalAction.SELL, symbol=symbol,
                sl_pips=self.sl_pips, tp_pips=self.tp_pips,
                confidence=conf, reason=reason,
            )

        return self._hold(symbol)

    # ------------------------------------------------------------------
    def _confidence(self, s: dict) -> float:
        """Blend trend strength (ADX) and breakout thrust into [0, 1]."""
        # ADX component: adx_min → 0.0, 40+ → 1.0
        adx_comp = max(0.0, min((s["adx"] - self.adx_min) / (40.0 - self.adx_min), 1.0))
        # Thrust component: how far past the breakout level vs ATR
        atr_v = s["atr"] if s["atr"] > 0 else 1e-9
        if s["ema_fast"] > s["ema_slow"]:
            thrust = (s["close"] - s["breakout_high"]) / atr_v
        else:
            thrust = (s["breakout_low"] - s["close"]) / atr_v
        thrust_comp = max(0.0, min(thrust, 1.0))
        return round(0.5 * adx_comp + 0.5 * thrust_comp, 3)

    # ------------------------------------------------------------------
    def get_proximity(self, df: pd.DataFrame, symbol: str) -> TriggerProximity:
        if len(df) < self._min_candles():
            return TriggerProximity()
        if "close" not in df:
            return TriggerProximity()

        s = self._compute(df)
        if any(pd.isna(v) for v in s.values()):
            return TriggerProximity()

        atr_v = s["atr"] if s["atr"] > 0 else 1e-9
        uptrend = s["ema_fast"] > s["ema_slow"]

        if uptrend:
            direction = "BUY"
            # distance below the breakout high, normalised by ATR
            dist = max(s["breakout_high"] - s["close"], 0.0)
            level = s["breakout_high"]
        else:
            direction = "SELL"
            dist = max(s["close"] - s["breakout_low"], 0.0)
            level = s["breakout_low"]

        # Proximity: 1.0 when price is at/through the breakout level AND ADX is
        # above the gate; fades as price sits further from the level or ADX is low.
        price_prox = max(0.0, 1.0 - dist / atr_v)
        adx_gate = 1.0 if s["adx"] >= self.adx_min else max(0.0, s["adx"] / self.adx_min)
        proximity = round(min(price_prox * adx_gate, 1.0), 4)

        return TriggerProximity(
            proximity_pct=proximity,
            direction=direction,
            label=f"{direction} brk {level:.5f} (ADX {s['adx']:.0f})",
            detail=(
                f"{dist:.5f} from {self.breakout_lookback}-bar "
                f"{'high' if uptrend else 'low'}; ADX {s['adx']:.1f}/"
                f"{self.adx_min:.0f}"
            ),
            indicator_values={
                "ema_fast": round(s["ema_fast"], 5),
                "ema_slow": round(s["ema_slow"], 5),
                "adx": round(s["adx"], 1),
                "breakout_high": round(s["breakout_high"], 5),
                "breakout_low": round(s["breakout_low"], 5),
            },
        )
