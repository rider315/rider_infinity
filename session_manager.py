"""
session_manager.py - Forex trading-session awareness.

Provides the ``SessionManager`` class that knows about Sydney, Tokyo,
London, and New York session hours, which pairs are best suited for
each session, and whether the market is currently open (forex trades
24/5, Sunday evening to Friday evening UTC).
"""

import logging
from datetime import datetime, timezone

logger = logging.getLogger("session_manager")


# Crypto bases that trade 24/7 (incl. weekends).  Detection is prefix-based so
# broker suffixes are tolerated: "BTCUSD", "BTCUSD.", "BTCUSDm", "ETHUSDT" …
CRYPTO_BASES: tuple[str, ...] = (
    "BTC", "ETH", "LTC", "XRP", "BCH", "ADA", "DOT", "SOL", "DOGE", "BNB",
    "LINK", "MATIC", "AVAX", "TRX", "XLM", "ATOM",
)


def is_crypto(symbol: str | None) -> bool:
    """Return ``True`` when *symbol* is a crypto pair (trades 24/7).

    Matches on the leading base ticker so broker-specific prefixes/suffixes
    (``BTCUSD``, ``BTCUSD.``, ``BTCUSDm``, ``Bitcoin``…) are tolerated.
    """
    if not symbol:
        return False
    s = symbol.upper().lstrip("#._-")
    if s.startswith("BITCOIN") or s.startswith("ETHEREUM"):
        return True
    return any(s.startswith(base) for base in CRYPTO_BASES)


class SessionManager:
    """Tracks forex trading sessions and recommends pairs accordingly.

    Session windows are defined in **UTC** and stored as class-level
    constants so they can be referenced without instantiation.

    Usage::

        sm = SessionManager()
        print(sm.get_active_sessions())   # e.g. ['London', 'New York']
        print(sm.is_overlap())            # True during London/NY overlap
    """

    # ── Session definitions (UTC hours, start inclusive, end exclusive) ──
    SYDNEY:   tuple[int, int] = (22, 7)     # 22:00 – 07:00 UTC (wraps midnight)
    TOKYO:    tuple[int, int] = (0, 9)      # 00:00 – 09:00 UTC
    LONDON:   tuple[int, int] = (7, 16)     # 07:00 – 16:00 UTC
    NEW_YORK: tuple[int, int] = (12, 21)    # 12:00 – 21:00 UTC

    # Mapping of session names → (start_hour, end_hour) for iteration.
    _SESSIONS: dict[str, tuple[int, int]] = {
        "Sydney":   SYDNEY,
        "Tokyo":    TOKYO,
        "London":   LONDON,
        "New York": NEW_YORK,
    }

    # Best pairs per session (most liquid during that window).
    _BEST_PAIRS: dict[str, list[str]] = {
        "Sydney":   ["AUDUSD", "NZDUSD", "AUDNZD"],
        "Tokyo":    ["USDJPY", "AUDJPY", "EURJPY"],
        "London":   ["EURUSD", "GBPUSD", "EURGBP"],
        "New York": ["EURUSD", "GBPUSD", "USDCAD"],
    }

    # ── Public API ───────────────────────────────────────────────────────

    def get_active_sessions(self, now: datetime | None = None) -> list[str]:
        """Return the names of every session that is currently open.

        Args:
            now: Optional explicit UTC datetime (useful for testing).
                 Defaults to the current UTC time.

        Returns:
            A list of session names, e.g. ``["London", "New York"]``.
            May be empty outside of all session windows.
        """
        now = now or datetime.now(timezone.utc)
        hour = now.hour
        active = []
        for name, (start, end) in self._SESSIONS.items():
            if start <= end:
                # Normal range (e.g. 0–9, 7–16, 12–21)
                if start <= hour < end:
                    active.append(name)
            else:
                # Wraps midnight (e.g. Sydney 22–07)
                if hour >= start or hour < end:
                    active.append(name)
        logger.debug("Active sessions at %02d:%02d UTC: %s", hour, now.minute, active)
        return active

    def is_market_open(
        self, now: datetime | None = None, symbol: str | None = None
    ) -> bool:
        """Check whether the market is open right now for *symbol*.

        Crypto symbols (BTC, ETH, …) trade 24/7, so this always returns
        ``True`` for them.  For forex/metals, the FX market runs from roughly
        Sunday 22:00 UTC to Friday 22:00 UTC; this returns ``True`` on weekdays
        (Mon-Fri) when at least one major session is active and ``False`` on
        Saturday/Sunday.

        Args:
            now: Optional explicit UTC datetime for testing.
            symbol: Optional symbol; crypto pairs bypass the forex calendar.

        Returns:
            ``True`` if trading is possible, ``False`` otherwise.
        """
        now = now or datetime.now(timezone.utc)

        # Crypto never closes — skip the forex weekend/session calendar.
        if is_crypto(symbol):
            return True

        # Monday = 0 … Sunday = 6.  Market is closed Sat & Sun.
        if now.weekday() >= 5:  # Saturday (5) or Sunday (6)
            logger.debug("Market closed – weekend.")
            return False

        # During the week, at least one session is virtually always
        # active, but we double-check.
        if self.get_active_sessions(now):
            return True

        # Between ~21:00 and 00:00 UTC on weekdays there is a brief
        # "gap" where no major session is defined.  The interbank
        # market is still technically open, but liquidity is very thin.
        logger.debug("No major session active at %02d:%02d UTC.", now.hour, now.minute)
        return False

    def next_market_open(self, now: datetime | None = None) -> dict:
        """Calculate when the market next opens and which session starts first.

        Returns:
            A dict with keys: 'opens_at' (datetime), 'session' (str),
            'hours_until' (float), 'label' (str).
        """
        now = now or datetime.now(timezone.utc)
        from datetime import timedelta

        # If it's a weekend, jump to next Monday 00:00 UTC (Tokyo opens)
        wd = now.weekday()
        if wd == 5:  # Saturday -> Monday
            opens = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=2)
        elif wd == 6:  # Sunday -> Monday
            opens = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        else:
            # Weekday but no session active — find next session start
            hour = now.hour
            next_session = None
            next_start = None
            for name, (start, end) in sorted(self._SESSIONS.items(), key=lambda x: x[1][0]):
                if hour < start:
                    next_session = name
                    next_start = start
                    break
            if next_session:
                opens = now.replace(hour=next_start, minute=0, second=0, microsecond=0)
            else:
                # Past all sessions today — next is Tokyo tomorrow (or Monday if Friday)
                if wd == 4:  # Friday -> Monday
                    opens = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=3)
                else:
                    opens = now.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                next_session = "Tokyo"

            if next_session is None:
                next_session = "Tokyo"

        if 'next_session' not in dir():
            next_session = "Tokyo"

        delta = opens - now
        hours = delta.total_seconds() / 3600
        if hours < 1:
            label = f"{int(delta.total_seconds() / 60)}m"
        elif hours < 24:
            label = f"{hours:.1f}h"
        else:
            label = f"{hours / 24:.1f}d"

        return {
            "opens_at": opens,
            "session": next_session,
            "hours_until": round(hours, 2),
            "label": label,
        }

    def get_best_pairs_for_session(self, session: str) -> list[str]:
        """Return the recommended currency pairs for a given session.

        Args:
            session: One of ``"Tokyo"``, ``"London"``, or ``"New York"``
                     (case-sensitive).

        Returns:
            A list of symbol strings, e.g. ``["EURUSD", "GBPUSD", "USDCAD"]``.

        Raises:
            ValueError: If *session* is not a recognised name.
        """
        if session not in self._BEST_PAIRS:
            raise ValueError(
                f"Unknown session '{session}'.  "
                f"Choose from: {', '.join(self._BEST_PAIRS)}"
            )
        return list(self._BEST_PAIRS[session])

    def is_overlap(self, now: datetime | None = None) -> bool:
        """Return ``True`` during the London / New York session overlap.

        The overlap (12:00-16:00 UTC) is widely considered the most
        liquid and volatile window, making it the best time for
        intraday strategies.

        Args:
            now: Optional explicit UTC datetime for testing.

        Returns:
            ``True`` if both London and New York sessions are active.
        """
        now = now or datetime.now(timezone.utc)
        hour = now.hour

        london_active   = self.LONDON[0]   <= hour < self.LONDON[1]
        newyork_active  = self.NEW_YORK[0] <= hour < self.NEW_YORK[1]

        overlap = london_active and newyork_active
        if overlap:
            logger.debug("London/NY overlap is active.")
        return overlap
