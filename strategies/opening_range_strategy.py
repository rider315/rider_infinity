"""
Opening-Range Breakout Strategy — Session Volatility Expansion
===============================================================

Volatility is serially predictable even when returns are not (volatility
clustering / GARCH).  A contraction preceding a major session open stores
energy; the London/New York open injects liquidity and information, which
releases directionally.  This strategy positions for that release by
trading a clean break of the session's opening range, optionally requiring
a prior volatility squeeze.

Signal logic
------------
* **BUY**  — within the trade window after a session open, the current close
  breaks *above* the opening-range high plus a buffer.
* **SELL** — close breaks *below* the opening-range low minus a buffer.
* **HOLD** — before the range forms, after the window closes, or no break.

The buffer (a fraction of ATR) keeps us out of marginal pokes that are
really stop-runs.  An optional squeeze pre-filter requires that volatility
was compressed as the range formed.

Best universe: GBPUSD, XAUUSD, EURUSD on M5/M15.  Day strategy — the
central engine should flatten at session end (no overnight risk).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .base import BaseStrategy, Signal, SignalAction, TriggerProximity
from .indicators import atr, bollinger_bandwidth

logger = logging.getLogger(__name__)


class OpeningRangeBreakoutStrategy(BaseStrategy):
    """
    Session opening-range breakout with optional volatility-squeeze filter.

    Parameters
    ----------
    session_opens : dict[str, int]
        UTC hour each tradable session opens (default London 07:00,
        New York 13:00).
    or_minutes : int
        Length of the opening range to build after the open (default 30).
    trade_window_hours : float
        How long after the open breakouts remain valid (default 1.5h —
        energy dissipates after that).
    buffer_atr : float
        Breakout buffer as a fraction of ATR (default 0.25).
    use_squeeze : bool
        Require the band-width at range-formation to be in the lower part of
        its recent distribution (default True).
    squeeze_factor : float
        Band-width must be ≤ ``squeeze_factor`` × its trailing-50 median
        (default 1.0 = at/below median).
    atr_period : int
        ATR lookback (default 14).
    sl_pips, tp_pips : float
        Default stop / target (≈1.8 R).
    """

    def __init__(
        self,
        session_opens: dict[str, int] | None = None,
        or_minutes: int = 30,
        trade_window_hours: float = 1.5,
        buffer_atr: float = 0.25,
        use_squeeze: bool = True,
        squeeze_factor: float = 1.0,
        atr_period: int = 14,
        sl_pips: float = 40,
        tp_pips: float = 72,
    ) -> None:
        super().__init__(sl_pips=sl_pips, tp_pips=tp_pips)
        self.session_opens = session_opens or {"LONDON": 7, "NY": 13}
        self.or_minutes = or_minutes
        self.trade_window_hours = trade_window_hours
        self.buffer_atr = buffer_atr
        self.use_squeeze = use_squeeze
        self.squeeze_factor = squeeze_factor
        self.atr_period = atr_period

    @property
    def name(self) -> str:
        return "Opening Range Breakout"

    def _active_session(self, ts: pd.Timestamp):
        """Return (session, open_timestamp) if ts is inside a trade window."""
        hour = ts.hour + ts.minute / 60.0
        day = ts.normalize()
        for sess, open_hour in self.session_opens.items():
            start = open_hour + self.or_minutes / 60.0
            end = open_hour + self.trade_window_hours
            if start <= hour <= end:
                return sess, day + pd.Timedelta(hours=open_hour)
        return None, None

    def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        if len(df) < self.atr_period + 25 or "time" not in df.columns:
            return self._hold(symbol)
        if not {"high", "low", "close"}.issubset(df.columns):
            return self._hold(symbol)

        ts = df["time"].iloc[-1]
        try:
            sess, open_ts = self._active_session(ts)
        except (AttributeError, TypeError):
            return self._hold(symbol)  # 'time' not a Timestamp
        if sess is None:
            return self._hold(symbol)

        or_end = open_ts + pd.Timedelta(minutes=self.or_minutes)
        mask = (df["time"] >= open_ts) & (df["time"] < or_end)
        positions = np.where(mask.to_numpy())[0]
        if len(positions) == 0:
            return self._hold(symbol)  # range not formed yet (e.g. weekend gap)

        or_bars = df.iloc[positions]
        orh = float(or_bars["high"].max())
        orl = float(or_bars["low"].min())

        atr_v = atr(df, self.atr_period).iloc[-1]
        if pd.isna(atr_v) or atr_v <= 0:
            return self._hold(symbol)

        # ---- Optional squeeze pre-filter (measured as the range formed) ----
        if self.use_squeeze:
            bw = bollinger_bandwidth(df["close"], 20, 2.0)
            last_or_pos = int(positions[-1])
            ref = bw.iloc[max(0, last_or_pos - 50):last_or_pos + 1].median()
            here = bw.iloc[last_or_pos]
            if pd.isna(here) or pd.isna(ref) or here > self.squeeze_factor * ref:
                logger.debug("[%s] %s — no squeeze, skipping %s OR.",
                             self.name, symbol, sess)
                return self._hold(symbol)

        buf = self.buffer_atr * atr_v
        close = float(df["close"].iloc[-1])

        if close > orh + buf:
            confidence = min((close - orh) / atr_v, 1.0)
            reason = (f"{sess} OR breakout: close={close:.5f} > ORH={orh:.5f}+buf, "
                      f"conf={confidence:.2f}")
            logger.info("[%s] %s — %s", self.name, symbol, reason)
            return Signal(SignalAction.BUY, symbol, self.sl_pips, self.tp_pips,
                          confidence, reason)

        if close < orl - buf:
            confidence = min((orl - close) / atr_v, 1.0)
            reason = (f"{sess} OR breakdown: close={close:.5f} < ORL={orl:.5f}-buf, "
                      f"conf={confidence:.2f}")
            logger.info("[%s] %s — %s", self.name, symbol, reason)
            return Signal(SignalAction.SELL, symbol, self.sl_pips, self.tp_pips,
                          confidence, reason)

        logger.debug("[%s] %s — %s: inside OR [%.5f, %.5f].",
                     self.name, symbol, sess, orl, orh)
        return self._hold(symbol)

    def get_proximity(self, df: pd.DataFrame, symbol: str) -> TriggerProximity:
        if len(df) < self.atr_period + 25 or "time" not in df.columns:
            return TriggerProximity()
        ts = df["time"].iloc[-1]
        try:
            sess, open_ts = self._active_session(ts)
        except (AttributeError, TypeError):
            return TriggerProximity()
        if sess is None:
            return TriggerProximity()

        or_end = open_ts + pd.Timedelta(minutes=self.or_minutes)
        mask = (df["time"] >= open_ts) & (df["time"] < or_end)
        if not mask.any():
            return TriggerProximity()

        or_bars = df[mask]
        orh, orl = float(or_bars["high"].max()), float(or_bars["low"].min())
        atr_v = atr(df, self.atr_period).iloc[-1]
        close = float(df["close"].iloc[-1])
        if pd.isna(atr_v) or atr_v <= 0:
            return TriggerProximity()

        dist_up = (orh - close) / atr_v
        dist_dn = (close - orl) / atr_v
        direction = "BUY" if dist_up <= dist_dn else "SELL"
        proximity = max(0.0, 1.0 - min(dist_up, dist_dn) / 1.0)
        return TriggerProximity(
            proximity_pct=round(min(proximity, 1.0), 4),
            direction=direction,
            label=f"{sess} OR [{orl:.5f}, {orh:.5f}]",
            detail=f"{min(dist_up, dist_dn):.2f} ATR from breakout",
            indicator_values={"or_high": round(orh, 6), "or_low": round(orl, 6),
                              "close": round(close, 6), "session": sess},
        )
