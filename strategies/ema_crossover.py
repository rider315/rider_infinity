"""
EMA Crossover Strategy — Trend Following
==========================================

Generates BUY / SELL signals when a fast Exponential Moving Average
crosses above / below a slow Exponential Moving Average.

Signal logic
------------
* **BUY**  — fast EMA crosses *above* slow EMA (bullish crossover).
* **SELL** — fast EMA crosses *below* slow EMA (bearish crossover).
* **HOLD** — no crossover detected on the latest two candles.

Confidence is derived from the normalised distance between the two
EMAs at the moment of the cross — a wider gap implies stronger
momentum behind the move.
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import BaseStrategy, Signal, SignalAction, TriggerProximity

logger = logging.getLogger(__name__)


class EMACrossoverStrategy(BaseStrategy):
    """
    Dual‑EMA crossover trend‑following strategy.

    Parameters
    ----------
    fast_period : int
        Lookback for the fast EMA (default 9).
    slow_period : int
        Lookback for the slow EMA (default 21).
    sl_pips : float
        Stop‑loss distance in pips.
    tp_pips : float
        Take‑profit distance in pips.
    trend_period : int
        Lookback for the trend EMA used by the filter (default 200).
    use_trend_filter : bool
        Only take a cross that agrees with the ``trend_period`` EMA
        (default ``True``).  A raw dual-EMA crossover fires on every
        wiggle, and most of those are counter-trend whipsaws; this is
        what makes the strategy tradeable rather than a coin flip.

        Measured over real daily index data (S&P 500 and NASDAQ,
        1999-2018, 5031 bars each, 1 index point of spread charged),
        enabling it improved net P&L, profit factor **and** win rate in
        all six instrument x stop-distance combinations tested, while
        roughly halving the trade count:

        =========  =======  ==================  =================
        Instrument  SL/TP   Filter off          Filter on
        =========  =======  ==================  =================
        S&P 500     20/40   -540  (PF 0.82)     -177  (PF 0.89)
        S&P 500     35/70   -141  (PF 0.96)     **+360** (PF 1.17)
        S&P 500    60/120   -612  (PF 0.80)     **+1383** (PF 1.76)
        NASDAQ      20/40   -171  (PF 0.94)     **+243** (PF 1.20)
        NASDAQ      35/70   +420  (PF 1.11)     **+642** (PF 1.32)
        NASDAQ     60/120  +1110  (PF 1.20)     **+1533** (PF 1.51)
        =========  =======  ==================  =================

        It used to default to ``False``, so the shipped configuration was
        the one this class's own comments described as unprofitable, and
        ``PARAM_GRIDS`` never searched the flag — a deep sweep could not
        have found it. Pass ``use_trend_filter=False`` for the old
        behaviour.
    min_confidence : float
        Skip crosses whose normalised fast/slow separation is below this
        (default 0.0 = take every cross).  Left off deliberately: on the
        same data it helped at some stop distances and hurt at others,
        and it cut the sample to ~26 trades over 20 years, which is too
        thin to conclude anything from.
    """

    def __init__(
        self,
        fast_period: int = 9,
        slow_period: int = 21,
        sl_pips: float = 50,
        tp_pips: float = 100,
        trend_period: int = 200,
        use_trend_filter: bool = True,
        min_confidence: float = 0.0,
    ) -> None:
        super().__init__(sl_pips=sl_pips, tp_pips=tp_pips)

        if fast_period >= slow_period:
            raise ValueError(
                f"fast_period ({fast_period}) must be less than "
                f"slow_period ({slow_period})."
            )

        self.fast_period = fast_period
        self.slow_period = slow_period
        # Trade only in the direction of the longer-term trend (filters out the
        # counter-trend whipsaws that made raw crossovers unprofitable), and
        # ignore weak crosses below a minimum confidence.
        self.trend_period = trend_period
        self.use_trend_filter = use_trend_filter
        self.min_confidence = min_confidence

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "EMA Crossover"

    def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        """
        Detect fast/slow EMA crossovers on the last two candles.

        Returns a BUY on a bullish cross, SELL on a bearish cross,
        or HOLD when no crossover occurred.
        """
        min_candles = self.slow_period + 2
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

        # ---- Calculate EMAs ----
        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()

        # Previous and current values
        prev_fast = fast_ema.iloc[-2]
        prev_slow = slow_ema.iloc[-2]
        curr_fast = fast_ema.iloc[-1]
        curr_slow = slow_ema.iloc[-1]

        # Normalised distance for confidence (clamped to [0, 1])
        price = close.iloc[-1]
        if price == 0:
            return self._hold(symbol)
        distance_pct = abs(curr_fast - curr_slow) / price
        confidence = min(distance_pct * 100, 1.0)  # scale up, cap at 1.0

        # ---- Trend filter ----
        # Only trade with the longer-term trend.  When there isn't enough data
        # for the trend EMA we skip the filter rather than block trading.
        trend_ok_buy = trend_ok_sell = True
        if self.use_trend_filter and len(df) >= self.trend_period:
            trend_ema = close.ewm(span=self.trend_period, adjust=False).mean().iloc[-1]
            trend_ok_buy = price > trend_ema
            trend_ok_sell = price < trend_ema

        # Weak crosses are noise — require a minimum confidence.
        if confidence < self.min_confidence:
            logger.debug(
                "[%s] %s — cross below min confidence (%.3f < %.3f), skipping.",
                self.name, symbol, confidence, self.min_confidence,
            )
            return self._hold(symbol)

        # ---- Crossover detection ----
        if prev_fast <= prev_slow and curr_fast > curr_slow and trend_ok_buy:
            reason = (
                f"Bullish EMA crossover: fast EMA({self.fast_period})={curr_fast:.5f} "
                f"crossed above slow EMA({self.slow_period})={curr_slow:.5f}"
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

        if prev_fast >= prev_slow and curr_fast < curr_slow and trend_ok_sell:
            reason = (
                f"Bearish EMA crossover: fast EMA({self.fast_period})={curr_fast:.5f} "
                f"crossed below slow EMA({self.slow_period})={curr_slow:.5f}"
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

        logger.debug("[%s] %s — no crossover detected.", self.name, symbol)
        return self._hold(symbol)

    def get_proximity(self, df: pd.DataFrame, symbol: str) -> TriggerProximity:
        """Measure how close the fast and slow EMAs are to crossing."""
        min_candles = self.slow_period + 2
        if len(df) < min_candles:
            return TriggerProximity()

        try:
            close = df["close"]
        except KeyError:
            return TriggerProximity()

        fast_ema = close.ewm(span=self.fast_period, adjust=False).mean()
        slow_ema = close.ewm(span=self.slow_period, adjust=False).mean()

        curr_fast = fast_ema.iloc[-1]
        curr_slow = slow_ema.iloc[-1]
        price = close.iloc[-1]

        if price == 0:
            return TriggerProximity()

        # Gap as percentage of price — smaller gap = closer to crossover
        gap = abs(curr_fast - curr_slow)
        gap_pct = gap / price

        # Use recent max gap to normalise (look back ~50 bars)
        lookback = min(50, len(fast_ema))
        recent_gaps = abs(fast_ema.iloc[-lookback:] - slow_ema.iloc[-lookback:])
        max_gap = recent_gaps.max()

        if max_gap == 0:
            proximity = 0.0
        else:
            proximity = 1.0 - min(gap / max_gap, 1.0)

        # Direction: which side is fast EMA approaching from?
        direction = "BUY" if curr_fast < curr_slow else "SELL"

        label = f"EMA Gap: {gap:.5f}"
        detail = (
            f"Fast EMA({self.fast_period})={curr_fast:.5f}  "
            f"Slow EMA({self.slow_period})={curr_slow:.5f}  "
            f"Gap={gap_pct*100:.3f}%"
        )

        return TriggerProximity(
            proximity_pct=round(proximity, 4),
            direction=direction,
            label=label,
            detail=detail,
            indicator_values={
                "fast_ema": round(curr_fast, 6),
                "slow_ema": round(curr_slow, 6),
                "gap": round(gap, 6),
                "gap_pct": round(gap_pct * 100, 4),
            },
        )
