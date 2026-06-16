"""
DCA (Dollar-Cost Averaging) Strategy
=====================================

Systematic position-building strategy that buys on dips below a
reference price (SMA) and takes profit when price recovers above
the reference by a target percentage.

Signal Logic
------------
* **BUY** — price is ``dip_threshold_pct`` below the SMA reference.
* **SELL** — price is ``take_profit_pct`` above the reference (take
  profit on existing positions).
* **HOLD** — price is within the neutral zone.
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import BaseStrategy, Signal, SignalAction, TriggerProximity

logger = logging.getLogger(__name__)


class DCAStrategy(BaseStrategy):
    """
    Dollar-Cost Averaging strategy.

    Parameters
    ----------
    dip_threshold_pct : float
        Percentage drop below reference to trigger a BUY (default 1.0).
    max_positions : int
        Maximum DCA entries allowed (default 5).
    take_profit_pct : float
        Percentage above reference to trigger a SELL / take-profit
        (default 2.0).
    reference_period : int
        SMA period used as the reference price (default 50).
    sl_pips : float
        Stop-loss in pips.
    tp_pips : float
        Take-profit in pips.
    """

    def __init__(
        self,
        dip_threshold_pct: float = 1.0,
        max_positions: int = 5,
        take_profit_pct: float = 2.0,
        reference_period: int = 50,
        sl_pips: float = 80.0,
        tp_pips: float = 60.0,
    ) -> None:
        super().__init__(sl_pips=sl_pips, tp_pips=tp_pips)
        self.dip_threshold_pct = dip_threshold_pct
        self.max_positions = max_positions
        self.take_profit_pct = take_profit_pct
        self.reference_period = reference_period

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "DCA (Dollar Cost Averaging)"

    def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Detect dip-buy or recovery-sell opportunities."""
        min_candles = self.reference_period + 5
        if len(df) < min_candles:
            return self._hold(symbol)

        try:
            close = df["close"]
        except KeyError:
            return self._hold(symbol)

        current_price = close.iloc[-1]
        reference = close.rolling(window=self.reference_period).mean().iloc[-1]

        if pd.isna(reference) or reference <= 0:
            return self._hold(symbol)

        # Percentage deviation from reference
        deviation_pct = ((current_price - reference) / reference) * 100.0

        # BUY on dip
        if deviation_pct <= -self.dip_threshold_pct:
            dip_depth = abs(deviation_pct)
            # Deeper dip = higher confidence (capped at 1.0)
            confidence = min(dip_depth / (self.dip_threshold_pct * 3), 1.0)
            reason = (
                f"DCA BUY: price {current_price:.5f} is "
                f"{dip_depth:.2f}% below SMA({self.reference_period})="
                f"{reference:.5f}"
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

        # SELL (take profit) on recovery
        if deviation_pct >= self.take_profit_pct:
            confidence = min(deviation_pct / (self.take_profit_pct * 2), 1.0)
            reason = (
                f"DCA SELL: price {current_price:.5f} is "
                f"{deviation_pct:.2f}% above SMA({self.reference_period})="
                f"{reference:.5f}"
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

        return self._hold(symbol)

    def get_proximity(self, df: pd.DataFrame, symbol: str) -> TriggerProximity:
        """Distance to the dip-buy threshold."""
        min_candles = self.reference_period + 5
        if len(df) < min_candles:
            return TriggerProximity()

        try:
            close = df["close"]
        except KeyError:
            return TriggerProximity()

        current_price = close.iloc[-1]
        reference = close.rolling(window=self.reference_period).mean().iloc[-1]
        if pd.isna(reference) or reference <= 0:
            return TriggerProximity()

        deviation_pct = ((current_price - reference) / reference) * 100.0

        if deviation_pct < 0:
            # Approaching dip threshold
            proximity = min(abs(deviation_pct) / self.dip_threshold_pct, 1.0)
            direction = "BUY"
            label = f"DCA: {abs(deviation_pct):.2f}% below ref"
            detail = f"{abs(deviation_pct):.2f}% / {self.dip_threshold_pct}% dip threshold"
        else:
            proximity = min(deviation_pct / self.take_profit_pct, 1.0)
            direction = "SELL"
            label = f"DCA: {deviation_pct:.2f}% above ref"
            detail = f"{deviation_pct:.2f}% / {self.take_profit_pct}% TP threshold"

        return TriggerProximity(
            proximity_pct=round(proximity, 4),
            direction=direction,
            label=label,
            detail=detail,
            indicator_values={
                "reference_price": round(reference, 5),
                "deviation_pct": round(deviation_pct, 2),
                "dip_threshold": self.dip_threshold_pct,
                "tp_threshold": self.take_profit_pct,
            },
        )

    def evaluate(self, df: pd.DataFrame, symbol: str) -> "EvalScore":
        """Return an evaluation score for the evaluation matrix."""
        from evaluation_matrix import EvalScore

        signal = self.analyze(df, symbol)

        if signal.action == SignalAction.BUY:
            value = -signal.confidence
        elif signal.action == SignalAction.SELL:
            value = signal.confidence
        else:
            value = 0.0

        return EvalScore(
            value=value,
            confidence=signal.confidence,
            weight=1.0,
            evaluator_name=self.name,
            reason=signal.reason,
            sl_pips=signal.sl_pips,
            tp_pips=signal.tp_pips,
        )
