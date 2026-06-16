"""
event_bus.py — Lightweight Pub/Sub Event Bus for Rider Infinity.

Provides a thread-safe, async-compatible publish/subscribe system that
serves as the communication backbone for all RI trading-bot components.

Components subscribe to named channels (market data, signals, orders,
risk alerts, etc.) and receive ``Event`` objects when something of
interest happens.  Both synchronous and asynchronous callbacks are
supported transparently.

Quick-start
-----------
>>> from event_bus import EventBus, EventChannel, emit, emit_sync
>>>
>>> bus = EventBus.get_instance()
>>>
>>> # Subscribe
>>> bus.subscribe(EventChannel.SIGNAL, my_callback)
>>>
>>> # Publish (async context)
>>> await emit(EventChannel.SIGNAL, "signal_generated",
...           {"symbol": "EURUSD", "action": "BUY"}, source="RSI")
>>>
>>> # Publish (sync context)
>>> emit_sync(EventChannel.ORDER, "order_filled",
...           {"ticket": 12345}, source="OrderManager")
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import threading
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Optional,
    Set,
    Union,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Event channels
# ---------------------------------------------------------------------------

class EventChannel(Enum):
    """Named channels that categorise events flowing through the bus.

    Each channel groups a family of related event types so that
    subscribers can opt-in to only the topics they care about.

    Attributes
    ----------
    MARKET_DATA : str
        Price ticks, candle closes, spread changes.
    SIGNAL : str
        Strategy evaluation scores and generated signals.
    TRADE_DECISION : str
        Aggregated trade decisions produced by the evaluation matrix.
    ORDER : str
        Order lifecycle events — placed, filled, cancelled, modified.
    RISK : str
        Risk-limit warnings, drawdown alerts, margin notifications.
    POSITION : str
        Position lifecycle — opened, closed, modified.
    AUTOMATION : str
        Automation trigger events (scheduled actions, rule firings).
    SYSTEM : str
        Bot lifecycle events — start, stop, error, reconnect.
    """

    MARKET_DATA = "market_data"
    SIGNAL = "signal"
    TRADE_DECISION = "trade_decision"
    ORDER = "order"
    RISK = "risk"
    POSITION = "position"
    AUTOMATION = "automation"
    SYSTEM = "system"


# ---------------------------------------------------------------------------
# Event value-object
# ---------------------------------------------------------------------------

@dataclass
class Event:
    """Immutable value-object that travels through the event bus.

    Attributes
    ----------
    channel : EventChannel
        The channel this event belongs to.
    event_type : str
        Specific event name, e.g. ``'candle_close'``,
        ``'signal_generated'``, ``'order_filled'``.
    data : dict
        Arbitrary payload dictionary.
    source : str
        Name of the component that emitted the event.
    timestamp : datetime
        UTC timestamp of when the event was created.  Defaults to
        ``datetime.now(timezone.utc)``.
    priority : int
        Urgency indicator.  ``0`` = normal, ``1`` = high,
        ``2`` = critical.
    """

    channel: EventChannel
    event_type: str
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: int = 0

    def __post_init__(self) -> None:
        if self.priority not in (0, 1, 2):
            raise ValueError(
                f"priority must be 0, 1, or 2 — got {self.priority!r}"
            )


# ---------------------------------------------------------------------------
# Subscriber wrapper
# ---------------------------------------------------------------------------

@dataclass
class _Subscriber:
    """Internal bookkeeping for a single subscription."""

    callback: Callable[..., Any]
    priority_filter: Optional[int] = None
    is_async: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_async = inspect.iscoroutinefunction(self.callback)

    def accepts(self, event: Event) -> bool:
        """Return ``True`` if this subscriber should receive *event*."""
        if self.priority_filter is not None:
            return event.priority >= self.priority_filter
        return True


# ---------------------------------------------------------------------------
# EventBus (singleton)
# ---------------------------------------------------------------------------

_MAX_HISTORY: int = 1000


class EventBus:
    """Thread-safe, async-compatible publish/subscribe event bus.

    Designed as a **singleton** — obtain the shared instance via
    ``EventBus.get_instance()``.  Direct construction is allowed for
    testing but discouraged in production code.

    Parameters
    ----------
    None

    Examples
    --------
    >>> bus = EventBus.get_instance()
    >>> bus.subscribe(EventChannel.ORDER, on_order_event)
    >>> await bus.publish(Event(
    ...     channel=EventChannel.ORDER,
    ...     event_type="order_filled",
    ...     data={"ticket": 42},
    ...     source="OrderManager",
    ... ))
    """

    _instance: Optional[EventBus] = None
    _instance_lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------ #
    # Construction / singleton
    # ------------------------------------------------------------------ #
    def __init__(self) -> None:
        self._lock: threading.Lock = threading.Lock()
        self._subscribers: Dict[EventChannel, List[_Subscriber]] = {
            ch: [] for ch in EventChannel
        }
        self._history: Deque[Event] = deque(maxlen=_MAX_HISTORY)

    @classmethod
    def get_instance(cls) -> EventBus:
        """Return the singleton ``EventBus`` instance.

        The instance is created lazily on first access and is
        protected by a lock for thread safety.

        Returns
        -------
        EventBus
            The shared event-bus instance.
        """
        if cls._instance is None:
            with cls._instance_lock:
                # Double-checked locking.
                if cls._instance is None:
                    cls._instance = cls()
                    logger.debug("EventBus singleton created.")
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Destroy the singleton and clear all subscriptions.

        Intended **only** for unit-test isolation — never call this in
        production code.
        """
        with cls._instance_lock:
            if cls._instance is not None:
                with cls._instance._lock:
                    cls._instance._subscribers = {
                        ch: [] for ch in EventChannel
                    }
                    cls._instance._history.clear()
                logger.debug("EventBus singleton reset — all subscriptions cleared.")
            cls._instance = None

    # ------------------------------------------------------------------ #
    # Subscribe / unsubscribe
    # ------------------------------------------------------------------ #
    def subscribe(
        self,
        channel: EventChannel,
        callback: Callable[..., Any],
        priority_filter: Optional[int] = None,
    ) -> None:
        """Register *callback* to receive events on *channel*.

        Parameters
        ----------
        channel : EventChannel
            The channel to subscribe to.
        callback : Callable
            A sync or async callable that accepts a single ``Event``
            argument.
        priority_filter : int, optional
            If set, the callback will only be invoked for events whose
            ``priority`` is **>=** this value.  For example, passing
            ``1`` means the callback ignores normal-priority events and
            only fires for high and critical ones.

        Raises
        ------
        TypeError
            If *callback* is not callable.
        """
        if not callable(callback):
            raise TypeError(
                f"callback must be callable, got {type(callback).__name__}"
            )

        sub = _Subscriber(callback=callback, priority_filter=priority_filter)
        with self._lock:
            # Prevent duplicate registrations of the same callback.
            for existing in self._subscribers[channel]:
                if existing.callback is callback:
                    logger.warning(
                        "Callback %r already subscribed to %s — skipping.",
                        callback,
                        channel.value,
                    )
                    return
            self._subscribers[channel].append(sub)

        logger.debug(
            "Subscribed %r to channel=%s (priority_filter=%s).",
            callback,
            channel.value,
            priority_filter,
        )

    def unsubscribe(
        self,
        channel: EventChannel,
        callback: Callable[..., Any],
    ) -> None:
        """Remove *callback* from *channel*.

        Parameters
        ----------
        channel : EventChannel
            The channel to unsubscribe from.
        callback : Callable
            The previously registered callback to remove.

        Notes
        -----
        Silently does nothing if *callback* was not subscribed.
        """
        with self._lock:
            before = len(self._subscribers[channel])
            self._subscribers[channel] = [
                s for s in self._subscribers[channel]
                if s.callback is not callback
            ]
            removed = before - len(self._subscribers[channel])

        if removed:
            logger.debug(
                "Unsubscribed %r from channel=%s.",
                callback,
                channel.value,
            )
        else:
            logger.debug(
                "Callback %r was not subscribed to channel=%s — no-op.",
                callback,
                channel.value,
            )

    # ------------------------------------------------------------------ #
    # Publish (async)
    # ------------------------------------------------------------------ #
    async def publish(self, event: Event) -> None:
        """Dispatch *event* to every matching subscriber (async version).

        Async callbacks are ``await``-ed; sync callbacks are called
        directly.  Each callback is wrapped in its own ``try``/
        ``except`` so that one failing subscriber never prevents
        others from receiving the event.

        Parameters
        ----------
        event : Event
            The event to dispatch.
        """
        self._record(event)

        with self._lock:
            subs = list(self._subscribers[event.channel])

        logger.debug(
            "Publishing %s.%s from '%s' (priority=%d) to %d subscriber(s).",
            event.channel.value,
            event.event_type,
            event.source,
            event.priority,
            len(subs),
        )

        for sub in subs:
            if not sub.accepts(event):
                continue
            try:
                if sub.is_async:
                    await sub.callback(event)
                else:
                    sub.callback(event)
            except Exception:
                logger.error(
                    "Subscriber %r raised on %s.%s — traceback follows.",
                    sub.callback,
                    event.channel.value,
                    event.event_type,
                    exc_info=True,
                )

    # ------------------------------------------------------------------ #
    # Publish (sync)
    # ------------------------------------------------------------------ #
    def publish_sync(self, event: Event) -> None:
        """Dispatch *event* synchronously — safe outside async context.

        If an ``asyncio`` event loop is already running in the current
        thread (e.g. inside a Jupyter notebook or an ``async`` server),
        the event is scheduled onto that loop via
        ``asyncio.ensure_future``.  Otherwise, a temporary loop is
        created to drive the dispatch.

        Parameters
        ----------
        event : Event
            The event to dispatch.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We are inside a running loop — schedule as a task.
            asyncio.ensure_future(self.publish(event))
        else:
            # No running loop — drive dispatch synchronously.
            self._publish_sync_inner(event)

    def _publish_sync_inner(self, event: Event) -> None:
        """Purely synchronous dispatch (no event-loop interaction).

        Async subscribers are skipped with a warning because there is
        no running event loop to drive them.

        Parameters
        ----------
        event : Event
            The event to dispatch.
        """
        self._record(event)

        with self._lock:
            subs = list(self._subscribers[event.channel])

        logger.debug(
            "Publishing (sync) %s.%s from '%s' (priority=%d) to %d subscriber(s).",
            event.channel.value,
            event.event_type,
            event.source,
            event.priority,
            len(subs),
        )

        for sub in subs:
            if not sub.accepts(event):
                continue
            try:
                if sub.is_async:
                    logger.warning(
                        "Async subscriber %r skipped during sync publish "
                        "on %s.%s — no running event loop.",
                        sub.callback,
                        event.channel.value,
                        event.event_type,
                    )
                    continue
                sub.callback(event)
            except Exception:
                logger.error(
                    "Subscriber %r raised on (sync) %s.%s — traceback follows.",
                    sub.callback,
                    event.channel.value,
                    event.event_type,
                    exc_info=True,
                )

    # ------------------------------------------------------------------ #
    # History
    # ------------------------------------------------------------------ #
    def _record(self, event: Event) -> None:
        """Append *event* to the rolling history buffer."""
        self._history.append(event)

    def get_event_history(
        self,
        channel: Optional[EventChannel] = None,
        limit: int = 100,
    ) -> List[Event]:
        """Return recent events from the history buffer.

        Parameters
        ----------
        channel : EventChannel, optional
            If given, only events on this channel are returned.
            ``None`` returns events across all channels.
        limit : int, optional
            Maximum number of events to return.  Defaults to ``100``.
            The buffer itself holds at most 1 000 events.

        Returns
        -------
        list[Event]
            Events ordered oldest-first.
        """
        with self._lock:
            if channel is not None:
                events = [e for e in self._history if e.channel is channel]
            else:
                events = list(self._history)
        return events[-limit:]

    # ------------------------------------------------------------------ #
    # Introspection helpers
    # ------------------------------------------------------------------ #
    def get_subscriber_count(
        self,
        channel: Optional[EventChannel] = None,
    ) -> int:
        """Return the number of active subscribers.

        Parameters
        ----------
        channel : EventChannel, optional
            If given, count only subscribers on that channel.
            ``None`` returns the total across all channels.

        Returns
        -------
        int
            Subscriber count.
        """
        with self._lock:
            if channel is not None:
                return len(self._subscribers[channel])
            return sum(len(v) for v in self._subscribers.values())


# ---------------------------------------------------------------------------
# Module-level convenience helpers
# ---------------------------------------------------------------------------

async def emit(
    channel: EventChannel,
    event_type: str,
    data: Dict[str, Any],
    source: str,
    priority: int = 0,
) -> None:
    """Create an ``Event`` and publish it asynchronously.

    This is a thin convenience wrapper around
    ``EventBus.get_instance().publish()``.

    Parameters
    ----------
    channel : EventChannel
        Target channel.
    event_type : str
        Descriptive event name (e.g. ``'candle_close'``).
    data : dict
        Payload dictionary.
    source : str
        Name of the emitting component.
    priority : int, optional
        ``0`` = normal, ``1`` = high, ``2`` = critical.
        Defaults to ``0``.

    Examples
    --------
    >>> await emit(
    ...     EventChannel.MARKET_DATA, "candle_close",
    ...     {"symbol": "EURUSD", "timeframe": "H1"},
    ...     source="MarketData",
    ... )
    """
    event = Event(
        channel=channel,
        event_type=event_type,
        data=data,
        source=source,
        priority=priority,
    )
    await EventBus.get_instance().publish(event)


def emit_sync(
    channel: EventChannel,
    event_type: str,
    data: Dict[str, Any],
    source: str,
    priority: int = 0,
) -> None:
    """Create an ``Event`` and publish it synchronously.

    Safe to call outside an ``async`` context.  Mirrors :func:`emit`
    but uses ``EventBus.publish_sync`` under the hood.

    Parameters
    ----------
    channel : EventChannel
        Target channel.
    event_type : str
        Descriptive event name.
    data : dict
        Payload dictionary.
    source : str
        Name of the emitting component.
    priority : int, optional
        ``0`` = normal, ``1`` = high, ``2`` = critical.
        Defaults to ``0``.

    Examples
    --------
    >>> emit_sync(
    ...     EventChannel.ORDER, "order_filled",
    ...     {"ticket": 12345, "symbol": "GBPUSD"},
    ...     source="OrderManager",
    ... )
    """
    event = Event(
        channel=channel,
        event_type=event_type,
        data=data,
        source=source,
        priority=priority,
    )
    EventBus.get_instance().publish_sync(event)
