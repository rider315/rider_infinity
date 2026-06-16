"""
triggers.py — Event Triggers for the Rider Infinity Automation Engine.

Triggers are the "entry point" of the Trigger → Condition → Action pipeline.
Each trigger monitors one aspect of the trading environment (price levels,
PnL thresholds, drawdown, time, volatility, etc.) and fires when its
criteria are met.

All concrete triggers inherit from :class:`BaseTrigger` and implement the
``check(context)`` method.  The base class provides **cooldown** logic so
a trigger cannot fire more often than ``cooldown_seconds`` allows, and
tracks the ``last_triggered`` timestamp.

Quick-start
-----------
>>> trigger = PriceThresholdTrigger("EURUSD", 1.1200, direction="above")
>>> ctx = {"prices": {"EURUSD": 1.1250}}
>>> trigger.check(ctx)
True
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base trigger
# ---------------------------------------------------------------------------

class BaseTrigger(ABC):
    """Abstract base class for all automation triggers.

    Provides cooldown gating, last-triggered tracking, and a ``reset()``
    method for re-arming one-shot triggers.

    Parameters
    ----------
    cooldown_seconds : int, optional
        Minimum elapsed time between consecutive firings.
        Defaults to ``300`` (5 minutes).

    Attributes
    ----------
    last_triggered : datetime or None
        UTC timestamp of the most recent firing, or *None* if the
        trigger has never fired.
    cooldown_seconds : int
        Minimum seconds between consecutive firings.
    """

    def __init__(self, cooldown_seconds: int = 300) -> None:
        self.cooldown_seconds: int = cooldown_seconds
        self.last_triggered: datetime | None = None

    # ------------------------------------------------------------------ #
    # Abstract interface
    # ------------------------------------------------------------------ #

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable identifier for this trigger."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-line description of what this trigger monitors."""

    @abstractmethod
    def check(self, context: dict) -> bool:
        """Evaluate the trigger against the current *context*.

        Parameters
        ----------
        context : dict
            Live trading context supplied by the automation engine.

        Returns
        -------
        bool
            ``True`` when the trigger condition is met **and** the
            cooldown has elapsed.
        """

    # ------------------------------------------------------------------ #
    # Cooldown helpers
    # ------------------------------------------------------------------ #

    def _can_trigger(self) -> bool:
        """Return ``True`` if the cooldown period has elapsed.

        Returns
        -------
        bool
            ``True`` when enough time has passed since the last firing,
            or the trigger has never fired.
        """
        if self.last_triggered is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self.last_triggered).total_seconds()
        return elapsed >= self.cooldown_seconds

    def _mark_triggered(self) -> None:
        """Record the current UTC time as the last firing instant."""
        self.last_triggered = datetime.now(timezone.utc)

    # ------------------------------------------------------------------ #
    # Reset
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """Reset internal state so the trigger can fire again.

        Subclasses that track crossing state (e.g. price-crossing
        detection) should override this and call ``super().reset()``.
        """
        self.last_triggered = None
        logger.debug("Trigger '%s' reset.", self.name)

    # ------------------------------------------------------------------ #
    # Serialisation helpers
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Return a JSON-serialisable snapshot of this trigger.

        Returns
        -------
        dict
            Contains ``type``, ``name``, ``description``,
            ``cooldown_seconds``, and ``last_triggered``.
        """
        return {
            "type": type(self).__name__,
            "name": self.name,
            "description": self.description,
            "cooldown_seconds": self.cooldown_seconds,
            "last_triggered": (
                self.last_triggered.isoformat() if self.last_triggered else None
            ),
        }


# ---------------------------------------------------------------------------
# Concrete triggers
# ---------------------------------------------------------------------------

class PriceThresholdTrigger(BaseTrigger):
    """Fire when a symbol's price **crosses** a threshold.

    This is a *one-shot crossing* trigger: it fires on the transition
    from below → above (``direction='above'``) or above → below
    (``direction='below'``), not while the price stays on the
    triggered side.

    Parameters
    ----------
    symbol : str
        Trading symbol to watch, e.g. ``"EURUSD"``.
    threshold : float
        The price level to monitor.
    direction : {'above', 'below'}
        ``'above'`` fires when price crosses from below to above the
        threshold; ``'below'`` fires when it crosses from above to below.
    cooldown_seconds : int, optional
        Minimum seconds between consecutive firings.  Default ``300``.

    Notes
    -----
    Context must contain ``{'prices': {symbol: float}}``.
    """

    def __init__(
        self,
        symbol: str,
        threshold: float,
        direction: str = "above",
        cooldown_seconds: int = 300,
    ) -> None:
        super().__init__(cooldown_seconds=cooldown_seconds)

        if direction not in ("above", "below"):
            raise ValueError(
                f"direction must be 'above' or 'below', got {direction!r}"
            )

        self.symbol: str = symbol
        self.threshold: float = threshold
        self.direction: str = direction
        self._previously_met: bool = False

    # -- Interface --------------------------------------------------------

    @property
    def name(self) -> str:
        return f"Price {self.direction.title()} {self.threshold} ({self.symbol})"

    @property
    def description(self) -> str:
        return (
            f"Fires when {self.symbol} price crosses "
            f"{self.direction} {self.threshold}"
        )

    def check(self, context: dict) -> bool:
        """Check for a price-level crossing.

        Parameters
        ----------
        context : dict
            Must include ``{'prices': {self.symbol: float}}``.

        Returns
        -------
        bool
        """
        prices = context.get("prices", {})
        current_price = prices.get(self.symbol)

        if current_price is None:
            return False

        if self.direction == "above":
            condition_met = current_price > self.threshold
        else:
            condition_met = current_price < self.threshold

        # One-shot crossing logic: fire only on the transition.
        triggered = condition_met and not self._previously_met and self._can_trigger()
        self._previously_met = condition_met

        if triggered:
            self._mark_triggered()
            logger.info(
                "PriceThresholdTrigger fired: %s price %.5f crossed %s %.5f",
                self.symbol,
                current_price,
                self.direction,
                self.threshold,
            )

        return triggered

    def reset(self) -> None:
        super().reset()
        self._previously_met = False

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "symbol": self.symbol,
            "threshold": self.threshold,
            "direction": self.direction,
        })
        return d


class PnLThresholdTrigger(BaseTrigger):
    """Fire when the daily PnL crosses a threshold.

    Uses one-shot crossing semantics identical to
    :class:`PriceThresholdTrigger`.

    Parameters
    ----------
    threshold : float
        PnL value to monitor (in account currency).
    direction : {'above', 'below'}
        ``'above'`` = profit target; ``'below'`` = loss limit.
    cooldown_seconds : int, optional
        Default ``300``.

    Notes
    -----
    Context must contain ``{'daily_pnl': float}``.
    """

    def __init__(
        self,
        threshold: float,
        direction: str = "above",
        cooldown_seconds: int = 300,
    ) -> None:
        super().__init__(cooldown_seconds=cooldown_seconds)

        if direction not in ("above", "below"):
            raise ValueError(
                f"direction must be 'above' or 'below', got {direction!r}"
            )

        self.threshold: float = threshold
        self.direction: str = direction
        self._previously_met: bool = False

    @property
    def name(self) -> str:
        return f"PnL {self.direction.title()} {self.threshold}"

    @property
    def description(self) -> str:
        return (
            f"Fires when daily PnL crosses "
            f"{self.direction} {self.threshold}"
        )

    def check(self, context: dict) -> bool:
        """Check for a daily-PnL crossing.

        Parameters
        ----------
        context : dict
            Must include ``{'daily_pnl': float}``.

        Returns
        -------
        bool
        """
        daily_pnl = context.get("daily_pnl")
        if daily_pnl is None:
            return False

        if self.direction == "above":
            condition_met = daily_pnl > self.threshold
        else:
            condition_met = daily_pnl < self.threshold

        triggered = condition_met and not self._previously_met and self._can_trigger()
        self._previously_met = condition_met

        if triggered:
            self._mark_triggered()
            logger.info(
                "PnLThresholdTrigger fired: daily PnL %.2f crossed %s %.2f",
                daily_pnl,
                self.direction,
                self.threshold,
            )

        return triggered

    def reset(self) -> None:
        super().reset()
        self._previously_met = False

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "threshold": self.threshold,
            "direction": self.direction,
        })
        return d


class DrawdownThresholdTrigger(BaseTrigger):
    """Fire when account drawdown exceeds a percentage limit.

    Uses one-shot crossing semantics: fires once when drawdown first
    exceeds the threshold, not on every tick while it stays above.

    Parameters
    ----------
    max_drawdown_pct : float
        Maximum acceptable drawdown percentage (e.g. ``5.0`` = 5 %).
    cooldown_seconds : int, optional
        Default ``300``.

    Notes
    -----
    Context must contain ``{'drawdown_pct': float}``.
    """

    def __init__(
        self,
        max_drawdown_pct: float,
        cooldown_seconds: int = 300,
    ) -> None:
        super().__init__(cooldown_seconds=cooldown_seconds)
        self.max_drawdown_pct: float = max_drawdown_pct
        self._previously_met: bool = False

    @property
    def name(self) -> str:
        return f"Drawdown > {self.max_drawdown_pct}%"

    @property
    def description(self) -> str:
        return (
            f"Fires when account drawdown exceeds "
            f"{self.max_drawdown_pct:.1f}%"
        )

    def check(self, context: dict) -> bool:
        """Check if drawdown has breached the threshold.

        Parameters
        ----------
        context : dict
            Must include ``{'drawdown_pct': float}``.

        Returns
        -------
        bool
        """
        drawdown_pct = context.get("drawdown_pct")
        if drawdown_pct is None:
            return False

        condition_met = drawdown_pct > self.max_drawdown_pct

        triggered = condition_met and not self._previously_met and self._can_trigger()
        self._previously_met = condition_met

        if triggered:
            self._mark_triggered()
            logger.warning(
                "DrawdownThresholdTrigger fired: drawdown %.2f%% > %.2f%%",
                drawdown_pct,
                self.max_drawdown_pct,
            )

        return triggered

    def reset(self) -> None:
        super().reset()
        self._previously_met = False

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"max_drawdown_pct": self.max_drawdown_pct})
        return d


class TimeBasedTrigger(BaseTrigger):
    """Fire once per day at a specific time.

    Parameters
    ----------
    target_hour : int
        Hour of day (0–23).
    target_minute : int, optional
        Minute of hour (0–59).  Default ``0``.
    tz_name : str, optional
        Timezone name.  Currently only ``'UTC'`` is supported.
        Default ``'UTC'``.
    cooldown_seconds : int, optional
        Default ``60``.

    Notes
    -----
    The trigger fires once per day within a ±30-second window around
    the target time.  After firing it will not fire again until the
    next calendar day (UTC).
    """

    def __init__(
        self,
        target_hour: int,
        target_minute: int = 0,
        tz_name: str = "UTC",
        cooldown_seconds: int = 60,
    ) -> None:
        super().__init__(cooldown_seconds=cooldown_seconds)
        self.target_hour: int = target_hour
        self.target_minute: int = target_minute
        self.tz_name: str = tz_name
        self._last_fire_date: datetime | None = None

    @property
    def name(self) -> str:
        return f"Time {self.target_hour:02d}:{self.target_minute:02d} {self.tz_name}"

    @property
    def description(self) -> str:
        return (
            f"Fires once per day at "
            f"{self.target_hour:02d}:{self.target_minute:02d} {self.tz_name}"
        )

    def check(self, context: dict) -> bool:
        """Check if the current UTC time matches the target.

        Parameters
        ----------
        context : dict
            Not used directly — time is read from the system clock.

        Returns
        -------
        bool
        """
        now = datetime.now(timezone.utc)

        # Already fired today?
        if (
            self._last_fire_date is not None
            and self._last_fire_date == now.date()
        ):
            return False

        # Within the target window?
        if now.hour != self.target_hour or now.minute != self.target_minute:
            return False

        if not self._can_trigger():
            return False

        self._mark_triggered()
        self._last_fire_date = now.date()
        logger.info(
            "TimeBasedTrigger fired at %s (target %02d:%02d %s).",
            now.strftime("%H:%M:%S"),
            self.target_hour,
            self.target_minute,
            self.tz_name,
        )
        return True

    def reset(self) -> None:
        super().reset()
        self._last_fire_date = None

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "target_hour": self.target_hour,
            "target_minute": self.target_minute,
            "tz_name": self.tz_name,
        })
        return d


class PeriodicTrigger(BaseTrigger):
    """Fire every *N* seconds.

    Useful for scheduled house-keeping tasks such as portfolio
    rebalancing or periodic risk checks.

    Parameters
    ----------
    interval_seconds : int
        The firing interval.

    Notes
    -----
    The ``cooldown_seconds`` parameter is set equal to
    ``interval_seconds`` so the base-class cooldown logic handles all
    timing.
    """

    def __init__(self, interval_seconds: int) -> None:
        super().__init__(cooldown_seconds=interval_seconds)
        self.interval_seconds: int = interval_seconds

    @property
    def name(self) -> str:
        return f"Every {self.interval_seconds}s"

    @property
    def description(self) -> str:
        return f"Fires every {self.interval_seconds} seconds"

    def check(self, context: dict) -> bool:
        """Always fires when the cooldown interval has elapsed.

        Parameters
        ----------
        context : dict
            Not used directly.

        Returns
        -------
        bool
        """
        if not self._can_trigger():
            return False

        self._mark_triggered()
        logger.debug(
            "PeriodicTrigger fired (interval=%ds).", self.interval_seconds
        )
        return True

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({"interval_seconds": self.interval_seconds})
        return d


class VolatilitySpikeTrigger(BaseTrigger):
    """Fire when current ATR significantly exceeds the rolling average.

    Detects sudden volatility expansions that may warrant pausing or
    adjusting trading behaviour.

    Parameters
    ----------
    atr_multiplier : float, optional
        Multiplier applied to the average ATR.  Default ``2.0``.
    lookback : int, optional
        Number of bars used for the average ATR (informational; the
        actual computation is external).  Default ``20``.
    cooldown_seconds : int, optional
        Default ``600`` (10 minutes).

    Notes
    -----
    Context must contain ``{'atr_current': float, 'atr_average': float}``.
    Uses one-shot crossing semantics.
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        lookback: int = 20,
        cooldown_seconds: int = 600,
    ) -> None:
        super().__init__(cooldown_seconds=cooldown_seconds)
        self.atr_multiplier: float = atr_multiplier
        self.lookback: int = lookback
        self._previously_met: bool = False

    @property
    def name(self) -> str:
        return f"Volatility Spike (ATR × {self.atr_multiplier})"

    @property
    def description(self) -> str:
        return (
            f"Fires when current ATR > average ATR × "
            f"{self.atr_multiplier} (lookback {self.lookback})"
        )

    def check(self, context: dict) -> bool:
        """Check for a volatility spike.

        Parameters
        ----------
        context : dict
            Must include ``{'atr_current': float, 'atr_average': float}``.

        Returns
        -------
        bool
        """
        atr_current = context.get("atr_current")
        atr_average = context.get("atr_average")

        if atr_current is None or atr_average is None:
            return False

        if atr_average <= 0:
            return False

        condition_met = atr_current > (atr_average * self.atr_multiplier)

        triggered = condition_met and not self._previously_met and self._can_trigger()
        self._previously_met = condition_met

        if triggered:
            self._mark_triggered()
            logger.warning(
                "VolatilitySpikeTrigger fired: ATR %.5f > avg %.5f × %.1f = %.5f",
                atr_current,
                atr_average,
                self.atr_multiplier,
                atr_average * self.atr_multiplier,
            )

        return triggered

    def reset(self) -> None:
        super().reset()
        self._previously_met = False

    def to_dict(self) -> dict:
        d = super().to_dict()
        d.update({
            "atr_multiplier": self.atr_multiplier,
            "lookback": self.lookback,
        })
        return d
