"""
trade_manager.py — Trade-management rules shared by every trading engine.

These are the rules that decide how much a trade may lose, when its stop
is pulled to safety, when a stale position is cut, and when a bot is
allowed to re-enter after being stopped out.  They were written for
``multi_bot.py`` (the fleet engine the dashboard drives) in response to
specific losses visible in the trade export, and they lived only there —
so ``bot.py``'s single-bot CLI loop kept trading with the behaviour they
were written to fix:

* no ATR-sized stop, so gold got the same stop distance as EURUSD;
* no stop clamp, so a strategy default could place an absurdly wide stop;
* no reward:risk floor, so a 55% win-rate still lost money because the
  average loss dwarfed the average win;
* no per-trade loss cap, no breakeven, no time-stop, and no post-loss
  cooldown, so losers ran to the full stop and the bot immediately
  re-entered the move that had just stopped it out.

Putting them here means both engines enforce one set of rules and a fix
cannot land in one path only.

Shape of a position
-------------------
The position-level helpers take plain dicts, the shape
``MultiBotManager._sync_position_state`` already builds.  Required keys:

====================  ======================================================
``ticket``            MT5 position ticket.
``type``              ``"BUY"`` or ``"SELL"``.
``entry``             Fill price.
``sl``                Current stop-loss price (``0.0`` if none).
``price_current``     Latest price for the closeable side.
``open_time``         Epoch seconds the position was opened.
====================  ======================================================

:func:`position_view` builds one from a raw MT5 position record.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Iterable, Optional, Sequence

try:
    import MetaTrader5 as mt5
except ImportError:  # pragma: no cover - keeps the module importable
    mt5 = None

from session_manager import is_crypto
from utils import compute_atr, pip_size, pips_to_price, price_to_pips

__all__ = [
    "DEAL_REASON_MAP",
    "apply_breakeven",
    "apply_time_stop",
    "breakeven_decisions",
    "cap_sl_for_quick_exit",
    "cooldown_block_reason",
    "cooldown_expiry",
    "pip_size",
    "pips_to_price",
    "position_view",
    "price_to_pips",
    "resolve_sl_tp",
    "should_start_cooldown",
    "stop_and_target_prices",
    "time_stop_decisions",
]

logger = logging.getLogger("trade_manager")

#: MT5 deal ``reason`` code → human-readable exit reason.  Used to tell a
#: stop-out from a manual close, which decides whether the post-loss
#: cooldown arms (see :func:`should_start_cooldown`).
DEAL_REASON_MAP: dict[int, str] = {
    0: "MANUAL",      # DEAL_REASON_CLIENT
    1: "MANUAL",      # DEAL_REASON_MOBILE
    2: "MANUAL",      # DEAL_REASON_WEB
    3: "EXPERT",      # DEAL_REASON_EXPERT
    4: "SL",          # DEAL_REASON_SL
    5: "TP",          # DEAL_REASON_TP
    6: "STOP_OUT",    # DEAL_REASON_SO
    7: "ROLLOVER",    # DEAL_REASON_ROLLOVER
}

#: Crypto stops are clamped to a fraction of price rather than to the
#: forex pip bounds — see :func:`resolve_sl_tp`.
_CRYPTO_MIN_SL_PCT = 0.0015   # 0.15% of price
_CRYPTO_MAX_SL_PCT = 0.015    # 1.5% of price
_CRYPTO_FALLBACK_SL_PCT = 0.004  # ~0.4% when ATR is unavailable


# ---------------------------------------------------------------------
# Position adapter
# ---------------------------------------------------------------------

def position_view(pos: Any) -> dict:
    """Convert a raw MT5 position record into the dict shape used here.

    Parameters
    ----------
    pos
        A record from ``mt5.positions_get()``.

    Returns
    -------
    dict
        The keys documented in this module's docstring, plus ``volume``,
        ``profit`` and ``swap`` for display.
    """
    buy_type = getattr(mt5, "ORDER_TYPE_BUY", 0) if mt5 is not None else 0
    return {
        "ticket": pos.ticket,
        "type": "BUY" if pos.type == buy_type else "SELL",
        "volume": pos.volume,
        "entry": pos.price_open,
        "price_current": pos.price_current,
        "sl": pos.sl,
        "tp": pos.tp,
        "profit": pos.profit,
        "swap": getattr(pos, "swap", 0.0),
        "open_time": getattr(pos, "time", 0),
    }


# ---------------------------------------------------------------------
# Stop / target sizing
# ---------------------------------------------------------------------

def resolve_sl_tp(
    config: Any, signal: Any, df: Any, symbol: str, point: float,
) -> tuple[float, float]:
    """Return ``(sl_pips, tp_pips)`` sized for volatility, clamped, with an RR floor.

    * If ``config.use_atr_stops`` is on and ATR is available, the stop is
      set to ``atr_sl_mult × ATR`` (converted to pips) — so volatile
      instruments like gold get wider stops and calm pairs get tighter ones.
    * The stop is clamped to ``[min_sl_pips, max_sl_pips]`` so it can never
      be absurdly wide (the root cause of the runaway losses).
    * The take-profit is forced to at least ``min_rr_ratio × sl_pips`` so a
      high win-rate can't be wrecked by tiny wins vs large losses.

    Crypto (BTC, ETH, …) is special-cased: the forex ``min/max_sl_pips``
    clamp is meaningless when 1 "pip" is a fraction of a dollar on a
    $60k instrument (a 60-pip stop would be a few dollars — an instant
    stop-out / broker "invalid stops" rejection). For crypto the stop is
    always ATR-driven and clamped to a **percentage-of-price** band so it
    scales with the instrument.

    Parameters
    ----------
    config
        The :class:`~config.Config` in force.
    signal
        Anything exposing ``sl_pips`` / ``tp_pips`` — normally a
        :class:`~strategies.base.Signal`.
    df
        OHLCV frame used for the ATR measurement.
    symbol
        Instrument name (decides the pip size and crypto handling).
    point
        The symbol's point size from ``symbol_info().point``.
    """
    rr = getattr(config, "min_rr_ratio", 1.8)
    # Baseline from the strategy's own suggestion (or global defaults).
    sl_pips = float(getattr(signal, "sl_pips", 0) or config.default_sl_pips)
    tp_pips = float(getattr(signal, "tp_pips", 0) or config.default_tp_pips)
    atr_based = False
    crypto = is_crypto(symbol)

    # Crypto always uses ATR stops (regardless of the global toggle); forex
    # only when use_atr_stops is on.
    if (crypto or getattr(config, "use_atr_stops", False)) and point > 0:
        atr_price = compute_atr(df, getattr(config, "atr_period", 14))
        if atr_price > 0:
            atr_pips = price_to_pips(
                atr_price * config.atr_sl_mult, symbol, point,
            )
            if atr_pips > 0:
                sl_pips = atr_pips
                atr_based = True

    if crypto and point > 0:
        # Clamp to a fraction-of-price band so the stop is neither dust nor
        # absurd. Fall back to a fixed percentage if ATR was unavailable.
        price = float(df["close"].iloc[-1])
        min_cx = price_to_pips(price * _CRYPTO_MIN_SL_PCT, symbol, point)
        max_cx = price_to_pips(price * _CRYPTO_MAX_SL_PCT, symbol, point)
        if not atr_based:
            sl_pips = price_to_pips(
                price * _CRYPTO_FALLBACK_SL_PCT, symbol, point,
            )
        sl_pips = max(min_cx, min(max_cx, sl_pips))
        atr_based = True  # always scale TP proportionally for crypto
    else:
        # Forex/metals: clamp the stop distance to sane pip bounds.
        sl_pips = max(config.min_sl_pips, min(config.max_sl_pips, sl_pips))

    if atr_based:
        # Volatility-based: scale TP proportionally so it's actually
        # reachable (a fixed far-away TP is why trades rarely hit profit).
        tp_pips = sl_pips * rr
    else:
        # Fixed-pip fallback: keep the suggested TP but never below the RR floor.
        tp_pips = max(tp_pips, sl_pips * rr)
    return round(sl_pips, 1), round(tp_pips, 1)


def cap_sl_for_quick_exit(
    config: Any, sl_pips: float, quick_exit_pips: float,
) -> tuple[float, Optional[str]]:
    """Clamp a Quick-Exit bot's stop to a multiple of its profit target.

    A bot that banks 5 pips must not risk a strategy's 50-pip default
    stop: risking 10× the reward needs a >90% win-rate to break even, and
    that exact pairing is what made the RSI bots net losers at a 73%
    win-rate.

    Returns
    -------
    tuple[float, str | None]
        The (possibly reduced) stop distance, and a human-readable note
        when it was reduced — ``None`` when it was left alone.
    """
    if quick_exit_pips <= 0:
        return sl_pips, None
    cap_mult = float(getattr(config, "quick_exit_sl_cap_mult", 0) or 0)
    if cap_mult <= 0:
        return sl_pips, None

    cap = max(
        quick_exit_pips * cap_mult,
        float(getattr(config, "min_sl_pips", 8)),
    )
    if sl_pips <= cap:
        return sl_pips, None

    note = (
        f"SL capped {sl_pips:.1f} -> {cap:.1f} pips "
        f"({cap_mult:.0f}x quick-exit target {quick_exit_pips:.1f})"
    )
    return round(cap, 1), note


# ---------------------------------------------------------------------
# Breakeven
# ---------------------------------------------------------------------

def breakeven_decisions(
    config: Any, positions: Iterable[dict],
) -> list[dict]:
    """Return the positions whose stop should be pulled to breakeven.

    This is the single highest-impact fix from the trade-export analysis.
    The account bled because losers ran to the **full** stop while winners
    were banked early — so the realised average loss (≈-$13) was far larger
    than the average win (≈+$8), turning a 55% win-rate into a net loss.

    For each open position, risk is ``R = |entry - SL|``.  Once floating
    profit reaches ``breakeven_trigger_r × R``, the stop should move to
    entry plus a small ``breakeven_lock_pips`` buffer.  A trade that first
    goes green and then reverses then exits at ≈breakeven instead of a
    full loss, which collapses the average loss and flips expectancy
    positive.

    Positions whose stop is already at or beyond breakeven are skipped, so
    the caller does not spam ``modify`` requests.

    Returns
    -------
    list[dict]
        One ``{"ticket", "offset_pips", "r_multiple"}`` per position to
        move.  Empty when the feature is disabled or nothing qualifies.
    """
    if not getattr(config, "breakeven_enabled", True):
        return []
    trigger_r = float(getattr(config, "breakeven_trigger_r", 0.6))
    if trigger_r <= 0:
        return []
    lock_pips = float(getattr(config, "breakeven_lock_pips", 1.0))

    decisions: list[dict] = []
    for pos in positions:
        entry = pos.get("entry", 0.0)
        sl = pos.get("sl", 0.0)
        current = pos.get("price_current", 0.0)
        ticket = pos.get("ticket")
        ptype = pos.get("type")
        if not entry or not current or not ticket or not sl:
            continue

        risk = abs(entry - sl)
        if risk <= 0:
            continue

        # Already locked at/beyond breakeven? Nothing to do.
        if ptype == "BUY" and sl >= entry:
            continue
        if ptype == "SELL" and sl <= entry:
            continue

        profit_price = (current - entry) if ptype == "BUY" else (entry - current)
        if profit_price >= trigger_r * risk:
            decisions.append({
                "ticket": ticket,
                "offset_pips": lock_pips,
                "r_multiple": profit_price / risk,
            })
    return decisions


def apply_breakeven(
    order_mgr: Any, config: Any, positions: Iterable[dict],
    log: logging.Logger, label: str = "",
) -> int:
    """Move qualifying stops to breakeven.  Returns the number moved."""
    moved = 0
    for decision in breakeven_decisions(config, positions):
        if order_mgr.set_breakeven(
            decision["ticket"], offset_pips=decision["offset_pips"],
        ):
            moved += 1
            log.info(
                "%sbreakeven set for #%s at +%.2fR (lock +%.1f pips).",
                f"{label} " if label else "",
                decision["ticket"], decision["r_multiple"],
                decision["offset_pips"],
            )
    return moved


# ---------------------------------------------------------------------
# Time-stop
# ---------------------------------------------------------------------

def time_stop_decisions(
    config: Any, positions: Iterable[dict], now: Optional[float] = None,
) -> list[dict]:
    """Return the positions held longer than ``config.max_trade_hours``.

    The export showed several trades held for days (e.g. a −$76 AUDJPY open
    for ~6 days). Those are stuck trades that will neither hit SL nor TP
    and just tie up risk and margin, so a hard maximum holding time caps
    that tail.  Disabled when ``max_trade_hours <= 0``.

    Returns
    -------
    list[dict]
        One ``{"ticket", "age_hours"}`` per position to close.
    """
    max_hours = float(getattr(config, "max_trade_hours", 0) or 0)
    if max_hours <= 0:
        return []
    now = time.time() if now is None else now

    decisions: list[dict] = []
    for pos in positions:
        opened = pos.get("open_time", 0)
        ticket = pos.get("ticket")
        if not opened or not ticket:
            continue
        age_hours = (now - opened) / 3600.0
        if age_hours >= max_hours:
            decisions.append({"ticket": ticket, "age_hours": age_hours})
    return decisions


def apply_time_stop(
    order_mgr: Any, config: Any, positions: Iterable[dict],
    log: logging.Logger, label: str = "", now: Optional[float] = None,
) -> int:
    """Close positions past the max holding time.  Returns the count closed."""
    max_hours = float(getattr(config, "max_trade_hours", 0) or 0)
    closed = 0
    for decision in time_stop_decisions(config, positions, now=now):
        prefix = f"{label} " if label else ""
        log.info(
            "%stime-stop closing #%s (held %.1fh >= %.1fh limit).",
            prefix, decision["ticket"], decision["age_hours"], max_hours,
        )
        if order_mgr.close_position(decision["ticket"]):
            closed += 1
        else:
            log.warning(
                "%stime-stop close failed for #%s - will retry.",
                prefix, decision["ticket"],
            )
    return closed


# ---------------------------------------------------------------------
# Post-loss cooldown
# ---------------------------------------------------------------------

def should_start_cooldown(
    config: Any, exit_reason: str, profit: float,
) -> bool:
    """Whether a closed trade should block re-entry for a while.

    A breakout signal stays valid for several candles, so without a
    cooldown the bot re-enters the same move seconds after being stopped
    out — the loss-churn that dominated the losses in the export (494 of
    954 XAUUSD trades opened <2 min after an SL, net −$330).

    Only a **losing** stop-out counts.  A winning SL exit (e.g. a trailed
    Quick-Exit stop) is a banked winner, not a failed entry.
    """
    if float(getattr(config, "sl_cooldown_min", 0) or 0) <= 0:
        return False
    return exit_reason in ("SL", "STOP_OUT") and profit < 0


def cooldown_expiry(config: Any, now: Optional[float] = None) -> float:
    """Return the epoch until which new entries are blocked."""
    minutes = float(getattr(config, "sl_cooldown_min", 0) or 0)
    now = time.time() if now is None else now
    return now + minutes * 60.0


def cooldown_block_reason(
    cooldown_until: float, now: Optional[float] = None,
) -> Optional[str]:
    """Return a human-readable block reason, or ``None`` if free to trade."""
    now = time.time() if now is None else now
    if now >= cooldown_until:
        return None
    remaining = int(cooldown_until - now)
    return (
        f"Post-loss cooldown ({remaining // 60}m{remaining % 60:02d}s left)"
    )


# ---------------------------------------------------------------------
# Entry price levels
# ---------------------------------------------------------------------

def stop_and_target_prices(
    direction: str, entry: float, sl_pips: float, tp_pips: float,
    symbol: str, point: float, digits: int,
) -> tuple[float, float]:
    """Convert pip distances into absolute SL/TP prices for an entry.

    Returns ``(sl_price, tp_price)`` rounded to the symbol's digits.
    """
    sl_delta = pips_to_price(sl_pips, symbol, point)
    tp_delta = pips_to_price(tp_pips, symbol, point)
    if direction == "BUY":
        return round(entry - sl_delta, digits), round(entry + tp_delta, digits)
    return round(entry + sl_delta, digits), round(entry - tp_delta, digits)
