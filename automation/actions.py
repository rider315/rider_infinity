"""
actions.py — Executable Actions for the Rider Infinity Automation Engine.

Actions are the final stage of the Trigger → Condition → Action pipeline.
When a trigger fires and all conditions pass, the attached actions execute
in order, each receiving the live trading *context* and returning an
:class:`ActionResult`.

Actions range from passive (logging / alerting) to highly impactful
(closing all positions, pausing trading).

Quick-start
-----------
>>> action = LogAction(message="Drawdown alert triggered", level="WARNING")
>>> result = action.execute(context={})
>>> result.success
True
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action result value-object
# ---------------------------------------------------------------------------

@dataclass
class ActionResult:
    """Outcome of an action execution.

    Attributes
    ----------
    success : bool
        ``True`` if the action completed without error.
    message : str
        Human-readable description of what happened.
    details : dict
        Optional structured data about the execution (e.g. closed
        ticket numbers, new parameter values).
    """

    success: bool
    message: str
    details: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Base action
# ---------------------------------------------------------------------------

class BaseAction(ABC):
    """Abstract base class for all automation actions.

    Each concrete action implements ``execute(context)`` and returns
    an :class:`ActionResult`.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable identifier for this action."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description of what this action does."""

    @abstractmethod
    def execute(self, context: dict) -> ActionResult:
        """Execute the action.

        Parameters
        ----------
        context : dict
            Live trading context.  Actions may both **read** context
            (e.g. to obtain an ``order_manager``) and **write** to it
            (e.g. to set a ``trading_paused`` flag).

        Returns
        -------
        ActionResult
            Outcome of the execution.
        """

    # ------------------------------------------------------------------ #
    # Serialisation helpers
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Return a JSON-serialisable snapshot of this action.

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
# Concrete actions
# ---------------------------------------------------------------------------

class PauseTradingAction(BaseAction):
    """Pause trading for a specified duration.

    Blocks new entries by arming :data:`trade_manager.GATE`, the
    process-wide switch every trading loop checks before opening a
    position.  Positions already open are left alone — breakeven, the
    time-stop and Quick Exit keep managing them.

    ``context['trading_paused']`` and ``context['trading_paused_until']``
    are also set, for conditions and log consumers that inspect the
    context.  Those keys are informational only: the context dict is
    rebuilt on every evaluation, so it is the gate that actually holds
    the pause.  (Before the gate existed, those keys were the *only*
    effect this action had, and nothing read them — so the Emergency Stop
    rule closed every position and then let the bots re-enter at once.)

    Parameters
    ----------
    duration_minutes : int, optional
        How long to pause trading.  Default ``30``.
    """

    def __init__(self, duration_minutes: int = 30) -> None:
        self.duration_minutes: int = duration_minutes

    @property
    def name(self) -> str:
        return f"Pause Trading ({self.duration_minutes} min)"

    @property
    def description(self) -> str:
        return f"Pauses trading for {self.duration_minutes} minutes"

    def execute(self, context: dict) -> ActionResult:
        """Arm the global trading gate and mirror the state into *context*.

        Parameters
        ----------
        context : dict
            Mutated to include ``trading_paused`` and
            ``trading_paused_until`` for inspection; the authoritative
            state lives in :data:`trade_manager.GATE`.

        Returns
        -------
        ActionResult
        """
        import trade_manager

        expiry = trade_manager.GATE.pause(
            self.duration_minutes,
            reason=context.get("pause_reason", "automation rule"),
        )

        context["trading_paused"] = True
        context["trading_paused_until"] = expiry.isoformat()

        msg = (
            f"Trading paused for {self.duration_minutes} minutes "
            f"until {expiry.strftime('%H:%M:%S UTC')}"
        )
        logger.warning(msg)

        return ActionResult(
            success=True,
            message=msg,
            details={
                "duration_minutes": self.duration_minutes,
                "paused_until": expiry.isoformat(),
            },
        )

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"duration_minutes": self.duration_minutes})
        return d


class CloseAllPositionsAction(BaseAction):
    """Close all open positions (optionally filtered by symbol).

    Requires ``context['order_manager']`` to be an instance of
    :class:`~order_manager.OrderManager`.

    Parameters
    ----------
    symbol : str or None, optional
        If specified, only positions for this symbol are closed.
        ``None`` closes every open position.  Default ``None``.
    """

    def __init__(self, symbol: str | None = None) -> None:
        self.symbol: str | None = symbol

    @property
    def name(self) -> str:
        sym_part = f" ({self.symbol})" if self.symbol else ""
        return f"Close All Positions{sym_part}"

    @property
    def description(self) -> str:
        if self.symbol:
            return f"Closes all open {self.symbol} positions"
        return "Closes all open positions across all symbols"

    def execute(self, context: dict) -> ActionResult:
        """Close positions via the order manager.

        Parameters
        ----------
        context : dict
            Must include ``{'order_manager': OrderManager}``.

        Returns
        -------
        ActionResult
        """
        order_manager = context.get("order_manager")

        if order_manager is None:
            msg = "CloseAllPositionsAction: no order_manager in context"
            logger.error(msg)
            return ActionResult(success=False, message=msg)

        try:
            closed_count = order_manager.close_all_positions(
                symbol=self.symbol
            )
            sym_desc = self.symbol or "all symbols"
            msg = f"Closed {closed_count} position(s) for {sym_desc}"
            logger.warning(msg)
            return ActionResult(
                success=True,
                message=msg,
                details={
                    "closed_count": closed_count,
                    "symbol": self.symbol,
                },
            )
        except Exception as exc:
            msg = f"CloseAllPositionsAction failed: {exc}"
            logger.error(msg, exc_info=True)
            return ActionResult(success=False, message=msg)

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"symbol": self.symbol})
        return d


class ReducePositionSizesAction(BaseAction):
    """Reduce future position sizes by a percentage.

    Sets ``context['lot_size_multiplier']`` to a value < 1.0 so that
    subsequent lot-size calculations produce smaller positions.

    Parameters
    ----------
    reduction_pct : float, optional
        Percentage to reduce sizes by (e.g. ``50.0`` means new
        positions will be 50 % of normal size).  Default ``50.0``.
    """

    def __init__(self, reduction_pct: float = 50.0) -> None:
        if not 0.0 < reduction_pct <= 100.0:
            raise ValueError(
                f"reduction_pct must be in (0, 100], got {reduction_pct}"
            )
        self.reduction_pct: float = reduction_pct

    @property
    def name(self) -> str:
        return f"Reduce Sizes by {self.reduction_pct}%"

    @property
    def description(self) -> str:
        remaining = 100.0 - self.reduction_pct
        return (
            f"Reduces future position sizes to {remaining:.0f}% "
            f"of normal"
        )

    def execute(self, context: dict) -> ActionResult:
        """Set the lot-size multiplier in *context*.

        Parameters
        ----------
        context : dict
            Will be mutated to include a ``lot_size_multiplier`` key.

        Returns
        -------
        ActionResult
        """
        multiplier = (100.0 - self.reduction_pct) / 100.0

        context["lot_size_multiplier"] = multiplier

        msg = (
            f"Lot size multiplier set to {multiplier:.2f} "
            f"({self.reduction_pct:.0f}% reduction)"
        )
        logger.warning(msg)

        return ActionResult(
            success=True,
            message=msg,
            details={
                "reduction_pct": self.reduction_pct,
                "multiplier": multiplier,
            },
        )

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"reduction_pct": self.reduction_pct})
        return d


class LogAction(BaseAction):
    """Log a message — useful for alerts without trading side-effects.

    Parameters
    ----------
    message : str
        The message to log.
    level : str, optional
        Log level name (``'DEBUG'``, ``'INFO'``, ``'WARNING'``,
        ``'ERROR'``, ``'CRITICAL'``).  Default ``'WARNING'``.
    """

    _LEVEL_MAP: dict[str, int] = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    def __init__(
        self,
        message: str,
        level: str = "WARNING",
    ) -> None:
        self.message: str = message
        self.level: str = level.upper()

        if self.level not in self._LEVEL_MAP:
            raise ValueError(
                f"level must be one of {list(self._LEVEL_MAP)}, "
                f"got {level!r}"
            )

    @property
    def name(self) -> str:
        return "Log Alert"

    @property
    def description(self) -> str:
        return f"Logs: '{self.message}' at {self.level} level"

    def execute(self, context: dict) -> ActionResult:
        """Emit the configured log message.

        Parameters
        ----------
        context : dict
            Not used directly.

        Returns
        -------
        ActionResult
        """
        numeric_level = self._LEVEL_MAP[self.level]
        logger.log(numeric_level, "[AUTOMATION] %s", self.message)

        return ActionResult(
            success=True,
            message=f"Logged at {self.level}: {self.message}",
            details={"level": self.level, "message": self.message},
        )

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"message": self.message, "level": self.level})
        return d
