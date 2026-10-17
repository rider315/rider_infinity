"""
Grid Trading Strategy
=====================

Profits from price oscillation within a range by placing buy orders
below a center price and sell orders above it.  No indicators needed —
the grid itself is the trading logic.

Signal Logic
------------
* **BUY** — price touches a buy-grid level (below center).
* **SELL** — price touches a sell-grid level (above center).
* **HOLD** — price is between grid levels or max orders reached.
"""

from __future__ import annotations

import logging


import pandas as pd

from instruments import estimate_pip_size

from evaluation_matrix import EvalScore

from .base import BaseStrategy, Signal, SignalAction, TriggerProximity


logger = logging.getLogger(__name__)


class GridStrategy(BaseStrategy):
    """
    Grid trading strategy — volatility harvesting.

    Parameters
    ----------
    grid_size_pips : float
        Distance between grid levels in pips (default 20).
    num_levels : int
        Number of levels above *and* below center (default 5).
    take_profit_pips : float
        Take-profit per grid order — usually equals grid_size (default 20).
    sl_pips : float
        Stop-loss in pips (default 40).
    tp_pips : float
        Global TP passed to BaseStrategy (default 20).
    max_open_orders : int
        Maximum concurrent grid orders (default 10).
    """

    def __init__(
        self,
        grid_size_pips: float = 20.0,
        num_levels: int = 5,
        max_open_orders: int = 10,
        sl_pips: float = 40.0,
        tp_pips: float = 20.0,
    ) -> None:
        super().__init__(sl_pips=sl_pips, tp_pips=tp_pips)
        self.grid_size_pips = grid_size_pips
        self.num_levels = num_levels
        self.max_open_orders = max_open_orders
        self._center_price: float | None = None

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "Grid Trading"

    def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Generate a signal when price touches a grid level."""
        min_candles = 50
        if len(df) < min_candles:
            return self._hold(symbol)

        try:
            close = df["close"]
        except KeyError:
            return self._hold(symbol)

        current_price = close.iloc[-1]

        # Auto-detect center from SMA(50)
        center = close.rolling(window=50).mean().iloc[-1]
        if pd.isna(center):
            return self._hold(symbol)
        self._center_price = center

        # Determine pip size
        pip_size = self._estimate_pip_size(symbol)

        # Calculate which grid level is nearest
        distance_pips = (current_price - center) / pip_size
        grid_position = round(distance_pips / self.grid_size_pips)

        # How close to the nearest grid level?
        nearest_level_pips = grid_position * self.grid_size_pips
        distance_to_level = abs(distance_pips - nearest_level_pips)

        # Only trigger if within 25% of a grid step to the level
        trigger_zone = self.grid_size_pips * 0.25

        if distance_to_level > trigger_zone:
            return self._hold(symbol)

        # Confidence: closer to level = higher confidence
        confidence = max(0.0, 1.0 - distance_to_level / trigger_zone)

        if grid_position < 0:
            # Below center → BUY
            level_price = center + grid_position * self.grid_size_pips * pip_size
            reason = (
                f"Grid BUY level {grid_position} (@{level_price:.5f}) "
                f"triggered at {current_price:.5f} (center={center:.5f})"
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

        elif grid_position > 0:
            # Above center → SELL
            level_price = center + grid_position * self.grid_size_pips * pip_size
            reason = (
                f"Grid SELL level +{grid_position} (@{level_price:.5f}) "
                f"triggered at {current_price:.5f} (center={center:.5f})"
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
        """Measure distance to the nearest grid level."""
        if len(df) < 50:
            return TriggerProximity()

        try:
            close = df["close"]
        except KeyError:
            return TriggerProximity()

        current_price = close.iloc[-1]
        center = close.rolling(window=50).mean().iloc[-1]
        if pd.isna(center):
            return TriggerProximity()

        pip_size = self._estimate_pip_size(symbol)
        distance_pips = (current_price - center) / pip_size
        grid_position = round(distance_pips / self.grid_size_pips)
        nearest_level_pips = grid_position * self.grid_size_pips
        dist_to_level = abs(distance_pips - nearest_level_pips)

        proximity = max(0.0, 1.0 - dist_to_level / self.grid_size_pips)
        direction = "BUY" if distance_pips < 0 else "SELL"

        return TriggerProximity(
            proximity_pct=round(proximity, 4),
            direction=direction,
            label=f"Grid: {dist_to_level:.1f} pips to level {grid_position}",
            detail=f"Center={center:.5f}, Grid level {grid_position}",
            indicator_values={
                "center_price": round(center, 5),
                "grid_position": grid_position,
                "dist_to_level_pips": round(dist_to_level, 2),
            },
        )

    def evaluate(self, df: pd.DataFrame, symbol: str) -> EvalScore:
        """Return an evaluation score for the evaluation matrix."""

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

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_pip_size(symbol: str) -> float:
        """Pip size for the grid's level spacing.

        A strategy never sees ``symbol_info()``, so this is an estimate
        from the symbol's family — see
        :func:`instruments.estimate_pip_size`. The previous inline
        version read gold's pip as 0.0001 instead of 0.1, mis-placing
        every grid level on metals by a factor of 1000.
        """
        return estimate_pip_size(symbol)
