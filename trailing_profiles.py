"""
trailing_profiles.py — Trailing Stop Profiles for the Rider Infinity Bot.

Provides a pluggable set of trailing stop strategies, inspired by
OctoBot's trailing stop architecture.  Each profile encapsulates a
distinct trailing logic (ATR-based, step-based, breakeven-first,
partial-close, time-decay) behind a common ``BaseTrailingProfile``
interface.

Usage::

    from trailing_profiles import get_trailing_profile, TrailingState

    profile = get_trailing_profile("atr", atr_multiplier=2.5)
    state   = TrailingState(ticket=123, symbol="EURUSD", direction="BUY",
                            entry_price=1.1000, current_sl=1.0950,
                            highest_price=1.1000)
    update  = profile.update(state, current_price=1.1080, current_atr=0.0012)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

from utils import pips_to_price, price_to_pips

logger = logging.getLogger(__name__)


# =====================================================================
# Value objects
# =====================================================================

@dataclass
class TrailingState:
    """Mutable state tracked per open position.

    Attributes
    ----------
    ticket : int
        MT5 position ticket.
    symbol : str
        Instrument name (e.g. ``"EURUSD"``).
    direction : str
        ``"BUY"`` or ``"SELL"``.
    entry_price : float
        Price at which the position was opened.
    current_sl : float
        Current stop-loss price on the position.
    highest_price : float
        Highest favourable price observed since entry.  For a BUY this
        is the highest high; for a SELL it is the lowest low.
    activated : bool
        Whether the trailing stop logic has been activated.
    partial_closed : bool
        Whether a partial close has already been executed (used by the
        :class:`PartialCloseTrailingStop` profile).
    breakeven_set : bool
        Whether breakeven has been applied (used by the
        :class:`BreakevenTrailingStop` profile).
    created_at : datetime
        UTC timestamp when this state was created.
    """

    ticket: int
    symbol: str
    direction: str
    entry_price: float
    current_sl: float
    highest_price: float
    activated: bool = False
    partial_closed: bool = False
    breakeven_set: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TrailingUpdate:
    """Result returned by :meth:`BaseTrailingProfile.update`.

    Attributes
    ----------
    new_sl : float | None
        Proposed new stop-loss price.  ``None`` means *no change*.
    activated : bool
        ``True`` if the activation state changed on this tick.
    partial_close : bool
        ``True`` when the profile wants a partial close executed.
    partial_close_pct : float
        Fraction of the position to close (``0.0``–``1.0``).
    reason : str
        Human-readable explanation for logging / journaling.
    """

    new_sl: float | None = None
    activated: bool = False
    partial_close: bool = False
    partial_close_pct: float = 0.0
    reason: str = ""


# =====================================================================
# Abstract base
# =====================================================================

class BaseTrailingProfile(ABC):
    """Abstract interface for all trailing stop profiles.

    Subclasses must implement :pyattr:`name` and :pymeth:`update`.
    The optional :pymeth:`should_activate` hook defaults to *always
    active* but can be overridden for profiles that require a minimum
    price excursion before trailing begins.

    Parameters
    ----------
    point : float
        Symbol point size (from ``symbol_info().point``).  Required for
        pip ↔ price conversions.
    """

    def __init__(self, point: float = 0.00001) -> None:
        self.point = point

    # -- Interface ------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable profile name."""
        ...

    @abstractmethod
    def update(
        self,
        state: TrailingState,
        current_price: float,
        current_atr: float = 0.0,
    ) -> TrailingUpdate:
        """Compute a new SL (if any) given current market conditions.

        Parameters
        ----------
        state : TrailingState
            Mutable position state.
        current_price : float
            Latest bid/ask price.
        current_atr : float
            Current ATR value (price units).  Only required by
            ATR-based profiles; others may ignore it.

        Returns
        -------
        TrailingUpdate
        """
        ...

    # -- Optional hooks -------------------------------------------------

    def should_activate(
        self, state: TrailingState, current_price: float
    ) -> bool:
        """Return ``True`` when trailing should start.

        Default behaviour: always activated.  Override for profiles
        that require a minimum move in the favourable direction.

        Parameters
        ----------
        state : TrailingState
            Current position state.
        current_price : float
            Latest bid/ask price.

        Returns
        -------
        bool
        """
        return True

    # -- Helpers --------------------------------------------------------

    def _pips_to_price(self, pips: float, symbol: str) -> float:
        """Shortcut for :func:`utils.pips_to_price`."""
        return pips_to_price(pips, symbol, self.point)

    def _price_to_pips(self, price_delta: float, symbol: str) -> float:
        """Shortcut for :func:`utils.price_to_pips`."""
        return price_to_pips(price_delta, symbol, self.point)

    def _favourable_delta(
        self, state: TrailingState, current_price: float
    ) -> float:
        """Return the favourable price move from entry (always ≥ 0)."""
        if state.direction == "BUY":
            return max(current_price - state.entry_price, 0.0)
        return max(state.entry_price - current_price, 0.0)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name!r}>"


# =====================================================================
# Concrete profiles
# =====================================================================

class ATRTrailingStop(BaseTrailingProfile):
    """Trail the stop-loss based on the current ATR value.

    The trailing distance adapts to market volatility — it widens in
    volatile conditions and tightens in calm ones.

    Parameters
    ----------
    atr_multiplier : float
        Multiplier applied to the ATR to compute trailing distance.
    activation_pips : float
        Minimum favourable excursion (in pips) before trailing begins.
    point : float
        Symbol point size.
    """

    def __init__(
        self,
        atr_multiplier: float = 2.0,
        activation_pips: float = 20.0,
        point: float = 0.00001,
    ) -> None:
        super().__init__(point=point)
        self.atr_multiplier = atr_multiplier
        self.activation_pips = activation_pips

    @property
    def name(self) -> str:  # noqa: D102
        return "ATR Trailing Stop"

    def should_activate(
        self, state: TrailingState, current_price: float
    ) -> bool:
        """Activate once price has moved *activation_pips* in our favour."""
        delta = self._favourable_delta(state, current_price)
        threshold = self._pips_to_price(self.activation_pips, state.symbol)
        return delta >= threshold

    def update(
        self,
        state: TrailingState,
        current_price: float,
        current_atr: float = 0.0,
    ) -> TrailingUpdate:
        # Check activation
        if not state.activated:
            if self.should_activate(state, current_price):
                logger.info(
                    "[%s] ATR trailing activated for ticket %d at %.5f",
                    state.symbol, state.ticket, current_price,
                )
                return TrailingUpdate(
                    activated=True,
                    reason=(
                        f"ATR trailing activated — price moved "
                        f"{self._price_to_pips(self._favourable_delta(state, current_price), state.symbol):.1f} "
                        f"pips in favour"
                    ),
                )
            return TrailingUpdate(reason="Waiting for activation")

        # Compute trail distance
        trail_distance = current_atr * self.atr_multiplier
        if trail_distance <= 0:
            logger.warning(
                "[%s] ATR is zero/negative (%.6f) for ticket %d — skipping",
                state.symbol, current_atr, state.ticket,
            )
            return TrailingUpdate(reason="ATR is zero — no update")

        # Proposed SL
        if state.direction == "BUY":
            proposed_sl = state.highest_price - trail_distance
            # Never move SL backwards
            if proposed_sl <= state.current_sl:
                return TrailingUpdate(
                    reason=(
                        f"BUY SL not advanced — proposed {proposed_sl:.5f} "
                        f"<= current {state.current_sl:.5f}"
                    ),
                )
        else:
            proposed_sl = state.highest_price + trail_distance
            if proposed_sl >= state.current_sl:
                return TrailingUpdate(
                    reason=(
                        f"SELL SL not advanced — proposed {proposed_sl:.5f} "
                        f">= current {state.current_sl:.5f}"
                    ),
                )

        move_pips = self._price_to_pips(
            abs(proposed_sl - state.current_sl), state.symbol,
        )
        logger.info(
            "[%s] ATR trail ticket %d: SL %.5f -> %.5f (%.1f pips, "
            "ATR=%.5f, mult=%.1f)",
            state.symbol, state.ticket, state.current_sl, proposed_sl,
            move_pips, current_atr, self.atr_multiplier,
        )
        return TrailingUpdate(
            new_sl=proposed_sl,
            reason=(
                f"ATR trail: SL moved {move_pips:.1f} pips "
                f"(ATR={current_atr:.5f} × {self.atr_multiplier})"
            ),
        )


# ----------------------------------------------------------------------

class StepTrailingStop(BaseTrailingProfile):
    """Move the stop-loss in fixed-size steps.

    The SL only advances when price has moved at least *step_pips*
    beyond the current SL, ensuring discrete, predictable ratcheting.

    Parameters
    ----------
    step_pips : float
        Size of each SL step in pips.
    activation_pips : float
        Minimum favourable excursion before trailing begins.
    point : float
        Symbol point size.
    """

    def __init__(
        self,
        step_pips: float = 20.0,
        activation_pips: float = 20.0,
        point: float = 0.00001,
    ) -> None:
        super().__init__(point=point)
        self.step_pips = step_pips
        self.activation_pips = activation_pips

    @property
    def name(self) -> str:  # noqa: D102
        return "Step Trailing Stop"

    def should_activate(
        self, state: TrailingState, current_price: float
    ) -> bool:
        delta = self._favourable_delta(state, current_price)
        threshold = self._pips_to_price(self.activation_pips, state.symbol)
        return delta >= threshold

    def update(
        self,
        state: TrailingState,
        current_price: float,
        current_atr: float = 0.0,
    ) -> TrailingUpdate:
        # Activation gate
        if not state.activated:
            if self.should_activate(state, current_price):
                logger.info(
                    "[%s] Step trailing activated for ticket %d at %.5f",
                    state.symbol, state.ticket, current_price,
                )
                return TrailingUpdate(
                    activated=True,
                    reason="Step trailing activated",
                )
            return TrailingUpdate(reason="Waiting for activation")

        step_price = self._pips_to_price(self.step_pips, state.symbol)

        if state.direction == "BUY":
            # How far price is beyond current SL
            gap = current_price - state.current_sl
            steps = int(gap / step_price) if step_price > 0 else 0
            if steps < 1:
                return TrailingUpdate(
                    reason="BUY — price has not moved a full step beyond SL",
                )
            # Move SL up by the number of complete steps, but cap so
            # SL never exceeds current price minus one step.
            proposed_sl = state.current_sl + steps * step_price
            max_sl = current_price - step_price
            proposed_sl = min(proposed_sl, max_sl)
            if proposed_sl <= state.current_sl:
                return TrailingUpdate(reason="Step SL not advanced for BUY")
        else:
            gap = state.current_sl - current_price
            steps = int(gap / step_price) if step_price > 0 else 0
            if steps < 1:
                return TrailingUpdate(
                    reason="SELL — price has not moved a full step beyond SL",
                )
            proposed_sl = state.current_sl - steps * step_price
            min_sl = current_price + step_price
            proposed_sl = max(proposed_sl, min_sl)
            if proposed_sl >= state.current_sl:
                return TrailingUpdate(reason="Step SL not advanced for SELL")

        move_pips = self._price_to_pips(
            abs(proposed_sl - state.current_sl), state.symbol,
        )
        logger.info(
            "[%s] Step trail ticket %d: SL %.5f -> %.5f (%d step(s), "
            "%.1f pips)",
            state.symbol, state.ticket, state.current_sl, proposed_sl,
            steps, move_pips,
        )
        return TrailingUpdate(
            new_sl=proposed_sl,
            reason=f"Step trail: SL moved {steps} step(s) ({move_pips:.1f} pips)",
        )


# ----------------------------------------------------------------------

class BreakevenTrailingStop(BaseTrailingProfile):
    """Two-phase profile: breakeven first, then standard trail.

    **Phase 1** — Wait until price is *breakeven_pips* in profit, then
    move SL to entry ± *buffer_pips*.

    **Phase 2** — Trail with a fixed *trail_pips* distance from the
    highest favourable price.

    Parameters
    ----------
    breakeven_pips : float
        Pips of profit required to trigger the breakeven move.
    trail_pips : float
        Trailing distance in pips once breakeven is set.
    buffer_pips : float
        Small buffer added beyond breakeven (protects against spread
        slippage).
    point : float
        Symbol point size.
    """

    def __init__(
        self,
        breakeven_pips: float = 20.0,
        trail_pips: float = 15.0,
        buffer_pips: float = 2.0,
        point: float = 0.00001,
    ) -> None:
        super().__init__(point=point)
        self.breakeven_pips = breakeven_pips
        self.trail_pips = trail_pips
        self.buffer_pips = buffer_pips

    @property
    def name(self) -> str:  # noqa: D102
        return "Breakeven Trailing Stop"

    def should_activate(
        self, state: TrailingState, current_price: float
    ) -> bool:
        """Activate once price reaches *breakeven_pips* profit."""
        delta = self._favourable_delta(state, current_price)
        threshold = self._pips_to_price(self.breakeven_pips, state.symbol)
        return delta >= threshold

    def update(
        self,
        state: TrailingState,
        current_price: float,
        current_atr: float = 0.0,
    ) -> TrailingUpdate:
        # ── Phase 1: move to breakeven ────────────────────────────────
        if not state.breakeven_set:
            if not self.should_activate(state, current_price):
                return TrailingUpdate(
                    reason=(
                        f"Waiting for {self.breakeven_pips} pips profit "
                        f"to set breakeven"
                    ),
                )

            buffer_price = self._pips_to_price(
                self.buffer_pips, state.symbol,
            )
            if state.direction == "BUY":
                proposed_sl = state.entry_price + buffer_price
            else:
                proposed_sl = state.entry_price - buffer_price

            logger.info(
                "[%s] Breakeven set for ticket %d: SL -> %.5f "
                "(entry=%.5f, buffer=%.1f pips)",
                state.symbol, state.ticket, proposed_sl,
                state.entry_price, self.buffer_pips,
            )
            return TrailingUpdate(
                new_sl=proposed_sl,
                activated=True,
                reason=(
                    f"Breakeven set at {proposed_sl:.5f} "
                    f"(entry + {self.buffer_pips} pip buffer)"
                ),
            )

        # ── Phase 2: standard trail ──────────────────────────────────
        trail_distance = self._pips_to_price(self.trail_pips, state.symbol)

        if state.direction == "BUY":
            proposed_sl = state.highest_price - trail_distance
            if proposed_sl <= state.current_sl:
                return TrailingUpdate(
                    reason="Breakeven trail: BUY SL not advanced",
                )
        else:
            proposed_sl = state.highest_price + trail_distance
            if proposed_sl >= state.current_sl:
                return TrailingUpdate(
                    reason="Breakeven trail: SELL SL not advanced",
                )

        move_pips = self._price_to_pips(
            abs(proposed_sl - state.current_sl), state.symbol,
        )
        logger.info(
            "[%s] Breakeven trail ticket %d: SL %.5f -> %.5f "
            "(%.1f pips, trail=%.1f pips)",
            state.symbol, state.ticket, state.current_sl, proposed_sl,
            move_pips, self.trail_pips,
        )
        return TrailingUpdate(
            new_sl=proposed_sl,
            reason=f"Breakeven trail: SL moved {move_pips:.1f} pips",
        )


# ----------------------------------------------------------------------

class PartialCloseTrailingStop(BaseTrailingProfile):
    """Close a portion of the position at a target, then trail the rest.

    At *target1_pips* profit the profile signals a partial close of
    *close_pct* of the position.  The remaining quantity is trailed
    with a fixed *trail_pips* distance.

    Parameters
    ----------
    target1_pips : float
        Profit target (in pips) at which the partial close fires.
    close_pct : float
        Fraction of the position to close (``0.0``–``1.0``).
    trail_pips : float
        Trailing distance for the remainder after partial close.
    point : float
        Symbol point size.
    """

    def __init__(
        self,
        target1_pips: float = 30.0,
        close_pct: float = 0.5,
        trail_pips: float = 20.0,
        point: float = 0.00001,
    ) -> None:
        super().__init__(point=point)
        self.target1_pips = target1_pips
        self.close_pct = max(0.0, min(close_pct, 1.0))
        self.trail_pips = trail_pips

    @property
    def name(self) -> str:  # noqa: D102
        return "Partial Close Trailing Stop"

    def should_activate(
        self, state: TrailingState, current_price: float
    ) -> bool:
        """Activate once price reaches *target1_pips* profit."""
        delta = self._favourable_delta(state, current_price)
        threshold = self._pips_to_price(self.target1_pips, state.symbol)
        return delta >= threshold

    def update(
        self,
        state: TrailingState,
        current_price: float,
        current_atr: float = 0.0,
    ) -> TrailingUpdate:
        # ── Check for partial close opportunity ───────────────────────
        if not state.partial_closed:
            if not self.should_activate(state, current_price):
                profit_pips = self._price_to_pips(
                    self._favourable_delta(state, current_price),
                    state.symbol,
                )
                return TrailingUpdate(
                    reason=(
                        f"Waiting for {self.target1_pips} pip target "
                        f"(current: {profit_pips:.1f} pips)"
                    ),
                )

            logger.info(
                "[%s] Partial close triggered for ticket %d: "
                "closing %.0f%% at %.5f",
                state.symbol, state.ticket,
                self.close_pct * 100, current_price,
            )
            return TrailingUpdate(
                activated=True,
                partial_close=True,
                partial_close_pct=self.close_pct,
                reason=(
                    f"Partial close {self.close_pct:.0%} at "
                    f"{self.target1_pips} pip target"
                ),
            )

        # ── Trail the remainder ───────────────────────────────────────
        trail_distance = self._pips_to_price(self.trail_pips, state.symbol)

        if state.direction == "BUY":
            proposed_sl = state.highest_price - trail_distance
            if proposed_sl <= state.current_sl:
                return TrailingUpdate(
                    reason="Partial-close trail: BUY SL not advanced",
                )
        else:
            proposed_sl = state.highest_price + trail_distance
            if proposed_sl >= state.current_sl:
                return TrailingUpdate(
                    reason="Partial-close trail: SELL SL not advanced",
                )

        move_pips = self._price_to_pips(
            abs(proposed_sl - state.current_sl), state.symbol,
        )
        logger.info(
            "[%s] Partial-close trail ticket %d: SL %.5f -> %.5f "
            "(%.1f pips)",
            state.symbol, state.ticket, state.current_sl, proposed_sl,
            move_pips,
        )
        return TrailingUpdate(
            new_sl=proposed_sl,
            reason=f"Partial-close trail: SL moved {move_pips:.1f} pips",
        )


# ----------------------------------------------------------------------

class TimeDecayTrailingStop(BaseTrailingProfile):
    """Tighten the trailing distance as the trade ages.

    The trail distance is linearly interpolated from *initial_distance*
    down to *min_distance* over *decay_hours*.  This prevents old
    winners from turning into losers.

    Parameters
    ----------
    initial_distance_pips : float
        Trailing distance at trade entry.
    min_distance_pips : float
        Minimum trailing distance (floor).
    decay_hours : float
        Hours over which the distance decays from initial to minimum.
    point : float
        Symbol point size.
    """

    def __init__(
        self,
        initial_distance_pips: float = 30.0,
        min_distance_pips: float = 10.0,
        decay_hours: float = 4.0,
        point: float = 0.00001,
    ) -> None:
        super().__init__(point=point)
        self.initial_distance_pips = initial_distance_pips
        self.min_distance_pips = min_distance_pips
        self.decay_hours = max(decay_hours, 0.001)  # avoid div-by-zero

    @property
    def name(self) -> str:  # noqa: D102
        return "Time Decay Trailing Stop"

    def _current_distance_pips(self, state: TrailingState) -> float:
        """Lerp between initial and minimum distance based on age."""
        now = datetime.now(timezone.utc)
        elapsed_seconds = (now - state.created_at).total_seconds()
        elapsed_hours = max(elapsed_seconds / 3600.0, 0.0)

        # Clamp t to [0, 1]
        t = min(elapsed_hours / self.decay_hours, 1.0)

        distance = (
            self.initial_distance_pips
            + t * (self.min_distance_pips - self.initial_distance_pips)
        )
        return distance

    def update(
        self,
        state: TrailingState,
        current_price: float,
        current_atr: float = 0.0,
    ) -> TrailingUpdate:
        # Always activated on first call
        activated_now = False
        if not state.activated:
            activated_now = True
            logger.info(
                "[%s] Time-decay trailing activated for ticket %d",
                state.symbol, state.ticket,
            )

        distance_pips = self._current_distance_pips(state)
        trail_distance = self._pips_to_price(distance_pips, state.symbol)

        if state.direction == "BUY":
            proposed_sl = state.highest_price - trail_distance
            if proposed_sl <= state.current_sl and not activated_now:
                return TrailingUpdate(
                    reason=(
                        f"Time-decay: BUY SL not advanced "
                        f"(distance={distance_pips:.1f} pips)"
                    ),
                )
        else:
            proposed_sl = state.highest_price + trail_distance
            if proposed_sl >= state.current_sl and not activated_now:
                return TrailingUpdate(
                    reason=(
                        f"Time-decay: SELL SL not advanced "
                        f"(distance={distance_pips:.1f} pips)"
                    ),
                )

        move_pips = self._price_to_pips(
            abs(proposed_sl - state.current_sl), state.symbol,
        )
        logger.info(
            "[%s] Time-decay trail ticket %d: SL %.5f -> %.5f "
            "(%.1f pips, distance=%.1f pips)",
            state.symbol, state.ticket, state.current_sl, proposed_sl,
            move_pips, distance_pips,
        )
        return TrailingUpdate(
            new_sl=proposed_sl,
            activated=activated_now,
            reason=(
                f"Time-decay trail: SL moved {move_pips:.1f} pips "
                f"(distance={distance_pips:.1f} pips)"
            ),
        )


# =====================================================================
# Registry & factory
# =====================================================================

TRAILING_PROFILE_REGISTRY: dict[str, type[BaseTrailingProfile]] = {
    "atr": ATRTrailingStop,
    "step": StepTrailingStop,
    "breakeven": BreakevenTrailingStop,
    "partial_close": PartialCloseTrailingStop,
    "time_decay": TimeDecayTrailingStop,
}

_PROFILE_DESCRIPTIONS: dict[str, str] = {
    "atr": (
        "ATR-based trailing stop — adapts to current market volatility."
    ),
    "step": (
        "Fixed-step trailing stop — ratchets SL in discrete pip increments."
    ),
    "breakeven": (
        "Breakeven-first trailing — moves to breakeven, then trails."
    ),
    "partial_close": (
        "Partial-close trailing — takes partial profit at a target, "
        "then trails the remainder."
    ),
    "time_decay": (
        "Time-decay trailing — tightens the trail distance as the "
        "trade ages."
    ),
}


def get_trailing_profile(
    name: str, **kwargs: object
) -> BaseTrailingProfile:
    """Factory function to instantiate a trailing profile by name.

    Parameters
    ----------
    name : str
        Registry key (e.g. ``"atr"``, ``"step"``).
    **kwargs
        Forwarded to the profile constructor.

    Returns
    -------
    BaseTrailingProfile
        A configured trailing profile instance.

    Raises
    ------
    ValueError
        If *name* is not found in the registry.
    """
    cls = TRAILING_PROFILE_REGISTRY.get(name)
    if cls is None:
        available = ", ".join(sorted(TRAILING_PROFILE_REGISTRY))
        raise ValueError(
            f"Unknown trailing profile {name!r}.  "
            f"Available profiles: {available}"
        )
    logger.info("Creating trailing profile %r with params %s", name, kwargs)
    return cls(**kwargs)  # type: ignore[arg-type]


def list_trailing_profiles() -> list[tuple[str, str]]:
    """Return a list of ``(name, description)`` tuples for all registered
    trailing profiles.

    Returns
    -------
    list[tuple[str, str]]
        Sorted list of available profiles.
    """
    return [
        (key, _PROFILE_DESCRIPTIONS.get(key, ""))
        for key in sorted(TRAILING_PROFILE_REGISTRY)
    ]
