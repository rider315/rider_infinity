"""
Base Strategy — Abstract Interface
====================================

Defines the contract every trading strategy must satisfy:
    * A ``name`` property for display / logging.
    * An ``analyze`` method that receives OHLCV candle data as a
      ``pandas.DataFrame`` and returns a ``Signal``.

All concrete strategies inherit from ``BaseStrategy``.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import pandas as pd

from evaluation_matrix import EvalScore



# ---------------------------------------------------------------------------
# Signal types
# ---------------------------------------------------------------------------

class SignalAction(Enum):
    """Possible actions emitted by a strategy."""

    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """
    Immutable value‑object returned by ``BaseStrategy.analyze``.

    Attributes
    ----------
    action : SignalAction
        The recommended action (BUY / SELL / HOLD).
    symbol : str
        The instrument symbol this signal refers to.
    sl_pips : float
        Suggested stop‑loss distance in pips.
    tp_pips : float
        Suggested take‑profit distance in pips.
    confidence : float
        Confidence score in the range ``[0.0, 1.0]``.
    reason : str
        Human‑readable explanation of *why* the signal was generated.
    """

    action: SignalAction
    symbol: str = ""
    sl_pips: float = 0.0
    tp_pips: float = 0.0
    confidence: float = 0.0  # 0.0 → 1.0
    reason: str = ""


@dataclass
class TriggerProximity:
    """Describes how close a strategy is to triggering a trade.

    Attributes
    ----------
    proximity_pct : float
        Percentage from 0.0 (no signal forming) to 1.0 (trigger imminent).
    direction : str
        Anticipated direction: ``"BUY"``, ``"SELL"``, or ``"NEUTRAL"``.
    label : str
        Short human-readable summary, e.g. ``"RSI: 34.2 → 30.0"``.
    detail : str
        Longer explanation, e.g. ``"4.2 points from oversold threshold"``.
    indicator_values : dict
        Raw indicator values for advanced UI rendering.
    """

    proximity_pct: float = 0.0
    direction: str = "NEUTRAL"
    label: str = "—"
    detail: str = ""
    indicator_values: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.

    Parameters
    ----------
    sl_pips : float
        Default stop‑loss distance in pips applied to every signal.
    tp_pips : float
        Default take‑profit distance in pips applied to every signal.
    """

    def __init__(self, sl_pips: float = 50, tp_pips: float = 100) -> None:
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips

    # -- Interface ----------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human‑readable strategy name (used in logs and UI)."""
        ...

    @abstractmethod
    def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        """
        Analyse OHLCV candle data and return a trading signal.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at least a ``close`` column.  Rows are ordered
            oldest → newest (index 0 is the oldest candle).
        symbol : str
            The instrument symbol, e.g. ``"EURUSD"``.

        Returns
        -------
        Signal
        """
        ...

    def get_proximity(self, df: pd.DataFrame, symbol: str) -> "TriggerProximity":
        """Return how close the strategy is to triggering a trade.

        Subclasses should override this to provide strategy-specific
        proximity metrics.  The default implementation returns a
        neutral zero-proximity result.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV candle data (same as ``analyze``).
        symbol : str
            The instrument symbol.

        Returns
        -------
        TriggerProximity
        """
        return TriggerProximity()

    def evaluate(self, df: pd.DataFrame, symbol: str) -> EvalScore:
        """Return an evaluation score for the evaluation matrix.

        Default implementation converts the ``analyze()`` Signal into an
        ``EvalScore``.  Subclasses can override for more nuanced scoring.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV candle data (same as ``analyze``).
        symbol : str
            The instrument symbol.

        Returns
        -------
        EvalScore
        """

        signal = self.analyze(df, symbol)

        if signal.action == SignalAction.BUY:
            value = -signal.confidence  # negative = bullish in OctoBot convention
        elif signal.action == SignalAction.SELL:
            value = signal.confidence   # positive = bearish
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

    # -- Helpers ------------------------------------------------------------

    def _hold(self, symbol: str = "") -> Signal:
        """Convenience factory for a neutral HOLD signal."""
        return Signal(
            action=SignalAction.HOLD,
            symbol=symbol,
            reason="No signal",
        )

    def __repr__(self) -> str:
        return (
            f"<{self.__class__.__name__} "
            f"name={self.name!r} sl={self.sl_pips} tp={self.tp_pips}>"
        )
