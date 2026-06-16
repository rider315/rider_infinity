"""
conditions.py — Gating Conditions for the Rider Infinity Automation Engine.

Conditions act as secondary guards in the Trigger → Condition → Action
pipeline.  After a trigger fires, **all** attached conditions must return
``True`` for the actions to execute.

This allows fine-grained control: for example, a drawdown trigger might
only run its actions during the London session and when a minimum number
of positions are open.

Quick-start
-----------
>>> cond = SessionActiveCondition(sessions=["london"])
>>> ctx = {"active_sessions": ["london", "new_york"]}
>>> cond.check(ctx)
True
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base condition
# ---------------------------------------------------------------------------

class BaseCondition(ABC):
    """Abstract base class for all automation conditions.

    Each condition inspects the live trading *context* dictionary and
    returns ``True`` (gate open) or ``False`` (gate closed).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable identifier for this condition."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description of what this condition checks."""

    @abstractmethod
    def check(self, context: dict) -> bool:
        """Evaluate the condition against the current *context*.

        Parameters
        ----------
        context : dict
            Live trading context supplied by the automation engine.

        Returns
        -------
        bool
            ``True`` if the condition is satisfied.
        """

    # ------------------------------------------------------------------ #
    # Serialisation helpers
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Return a JSON-serialisable snapshot of this condition.

        Returns
        -------
        dict
            Contains ``type``, ``name``, and ``description``.
        """
        return {
            "type": type(self).__name__,
            "name": self.name,
            "description": self.description,
        }


# ---------------------------------------------------------------------------
# Concrete conditions
# ---------------------------------------------------------------------------

class SessionActiveCondition(BaseCondition):
    """Pass when at least one of the specified trading sessions is active.

    Parameters
    ----------
    sessions : list[str]
        Session names to match against, e.g. ``['london', 'new_york']``.
        Matching is case-insensitive.

    Notes
    -----
    Context must contain ``{'active_sessions': list[str]}``.
    """

    def __init__(self, sessions: list[str]) -> None:
        self.sessions: list[str] = [s.lower() for s in sessions]

    @property
    def name(self) -> str:
        return f"Session Active ({', '.join(self.sessions)})"

    @property
    def description(self) -> str:
        return (
            f"Passes when any of [{', '.join(self.sessions)}] "
            f"sessions is active"
        )

    def check(self, context: dict) -> bool:
        """Check if any target session is currently active.

        Parameters
        ----------
        context : dict
            Must include ``{'active_sessions': list[str]}``.

        Returns
        -------
        bool
        """
        active = context.get("active_sessions", [])
        active_lower = [s.lower() for s in active]

        result = any(s in active_lower for s in self.sessions)

        logger.debug(
            "SessionActiveCondition: wanted=%s  active=%s  result=%s",
            self.sessions,
            active,
            result,
        )
        return result

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"sessions": self.sessions})
        return d


class MinPositionsCondition(BaseCondition):
    """Pass when the number of open positions meets a minimum threshold.

    Useful for rules that should only apply when the bot is actively
    holding trades (e.g. "reduce sizes when we have ≥ 3 positions").

    Parameters
    ----------
    min_count : int
        Minimum number of open positions required.

    Notes
    -----
    Context must contain ``{'open_positions_count': int}``.
    """

    def __init__(self, min_count: int) -> None:
        self.min_count: int = min_count

    @property
    def name(self) -> str:
        return f"Min Positions ≥ {self.min_count}"

    @property
    def description(self) -> str:
        return f"Passes when open position count ≥ {self.min_count}"

    def check(self, context: dict) -> bool:
        """Check the open position count.

        Parameters
        ----------
        context : dict
            Must include ``{'open_positions_count': int}``.

        Returns
        -------
        bool
        """
        count = context.get("open_positions_count", 0)
        result = count >= self.min_count

        logger.debug(
            "MinPositionsCondition: count=%d  min=%d  result=%s",
            count,
            self.min_count,
            result,
        )
        return result

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"min_count": self.min_count})
        return d


class MaxPositionsCondition(BaseCondition):
    """Pass when the number of open positions is at or below a maximum.

    Useful for rules that should only act when the bot still has room
    for more trades.

    Parameters
    ----------
    max_count : int
        Maximum number of open positions allowed.

    Notes
    -----
    Context must contain ``{'open_positions_count': int}``.
    """

    def __init__(self, max_count: int) -> None:
        self.max_count: int = max_count

    @property
    def name(self) -> str:
        return f"Max Positions ≤ {self.max_count}"

    @property
    def description(self) -> str:
        return f"Passes when open position count ≤ {self.max_count}"

    def check(self, context: dict) -> bool:
        """Check the open position count.

        Parameters
        ----------
        context : dict
            Must include ``{'open_positions_count': int}``.

        Returns
        -------
        bool
        """
        count = context.get("open_positions_count", 0)
        result = count <= self.max_count

        logger.debug(
            "MaxPositionsCondition: count=%d  max=%d  result=%s",
            count,
            self.max_count,
            result,
        )
        return result

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"max_count": self.max_count})
        return d


class SpreadBelowCondition(BaseCondition):
    """Pass when the current spread is within acceptable limits.

    Prevents actions from executing during wide-spread conditions
    (e.g. news events, low-liquidity periods).

    Parameters
    ----------
    max_spread_points : float
        Maximum acceptable spread in broker points.
    symbol : str or None, optional
        If specified, checks spread for this symbol only.  If ``None``,
        checks that **all** spreads in context are below the threshold.
        Default ``None``.

    Notes
    -----
    Context must contain ``{'spread_points': {symbol: float}}``.
    """

    def __init__(
        self,
        max_spread_points: float,
        symbol: str | None = None,
    ) -> None:
        self.max_spread_points: float = max_spread_points
        self.symbol: str | None = symbol

    @property
    def name(self) -> str:
        sym_part = f" ({self.symbol})" if self.symbol else ""
        return f"Spread ≤ {self.max_spread_points} pts{sym_part}"

    @property
    def description(self) -> str:
        sym_part = f" for {self.symbol}" if self.symbol else " for all symbols"
        return (
            f"Passes when spread ≤ {self.max_spread_points} points"
            f"{sym_part}"
        )

    def check(self, context: dict) -> bool:
        """Check if spreads are within acceptable limits.

        Parameters
        ----------
        context : dict
            Must include ``{'spread_points': {symbol: float}}``.

        Returns
        -------
        bool
        """
        spreads = context.get("spread_points", {})

        if not spreads:
            # No spread data available — conservatively pass.
            logger.debug(
                "SpreadBelowCondition: no spread data — passing by default."
            )
            return True

        if self.symbol is not None:
            spread = spreads.get(self.symbol)
            if spread is None:
                logger.debug(
                    "SpreadBelowCondition: no spread for %s — passing.",
                    self.symbol,
                )
                return True
            result = spread <= self.max_spread_points
        else:
            # All tracked symbols must pass.
            result = all(s <= self.max_spread_points for s in spreads.values())

        logger.debug(
            "SpreadBelowCondition: spreads=%s  max=%.1f  result=%s",
            spreads,
            self.max_spread_points,
            result,
        )
        return result

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "max_spread_points": self.max_spread_points,
            "symbol": self.symbol,
        })
        return d
