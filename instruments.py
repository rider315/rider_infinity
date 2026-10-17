"""
instruments.py — Instrument classification and pip arithmetic.

The single source of truth for "what is one pip on this symbol" and for
the symbol-family tests (crypto, metal) that question depends on.

Why this module exists
----------------------
Pip size was previously computed inline in six places, by five mutually
inconsistent rules:

==========================================  ==========================================
``utils.pips_to_price``                     ``100 × point`` for JPY, else ``10 × point``
``risk_manager`` (lot sizing, loss cap)     ``10 × point`` at 3/5 digits, else ``point``
``order_manager.set_breakeven``             same digits rule
``backtester``                              the JPY rule
``cloud/worker/worker_backtest``            hard-coded ``0.01`` / ``0.0001``, ignoring point
==========================================  ==========================================

They disagreed by 10× on gold and on every 3-digit JPY pair, and by
1000× on gold in the cloud worker.  Because the risk layer sized lots
with one rule while the order layer placed stops with another, a trade's
real risk did not match the risk it was sized for.  Keeping one
definition here — imported everywhere, dependency-free so the cloud
worker can ship it — is what stops that drifting apart again.

This module imports nothing outside the standard library, so it is safe
to import from any layer, including the standalone cloud worker.
"""

from __future__ import annotations

import math

# ── Symbol families ──────────────────────────────────────────────────

#: Crypto bases that trade 24/7 (incl. weekends).  Detection is
#: prefix-based so broker suffixes are tolerated: "BTCUSD", "BTCUSD.",
#: "BTCUSDm", "ETHUSDT" …
CRYPTO_BASES: tuple[str, ...] = (
    "BTC", "ETH", "LTC", "XRP", "BCH", "ADA", "DOT", "SOL", "DOGE", "BNB",
    "LINK", "MATIC", "AVAX", "TRX", "XLM", "ATOM",
)

#: Metal bases.  Market convention prices a metal to 2 decimals and calls
#: 0.1 one pip (gold moves "10 pips" per dollar), which the generic
#: digits rule would otherwise read as 0.01.
METAL_BASES: tuple[str, ...] = ("XAU", "XAG", "XPT", "XPD")

#: Characters brokers prepend to a ticker.
_LEADING_JUNK = "#._- "


def _normalise(symbol: str | None) -> str:
    """Upper-case *symbol* and strip broker decoration from the front."""
    if not symbol:
        return ""
    return symbol.upper().lstrip(_LEADING_JUNK)


def is_crypto(symbol: str | None) -> bool:
    """Return ``True`` when *symbol* is a crypto pair (trades 24/7).

    Matches on the leading base ticker so broker-specific prefixes and
    suffixes (``BTCUSD``, ``BTCUSD.``, ``BTCUSDm``, ``Bitcoin``…) are
    tolerated.
    """
    s = _normalise(symbol)
    if not s:
        return False
    if s.startswith("BITCOIN") or s.startswith("ETHEREUM"):
        return True
    return s.startswith(CRYPTO_BASES)


def is_metal(symbol: str | None) -> bool:
    """Return ``True`` when *symbol* is a precious metal (XAU, XAG, …)."""
    s = _normalise(symbol)
    return bool(s) and s.startswith(METAL_BASES)


# ── Pip arithmetic ───────────────────────────────────────────────────

def digits_from_point(point: float) -> int:
    """Recover an instrument's quote precision from its point size.

    ``point == 10 ** -digits``, and callers generally only have
    ``symbol_info().point`` to hand.
    """
    if point <= 0:
        return 0
    return int(round(-math.log10(point)))


def pip_size(symbol: str, point: float) -> float:
    """Return the price value of one pip for *symbol*.

    One pip is a quote convention, not a fixed number of points, and it
    follows the instrument's quote precision:

    ==================  ======  =========  ===========================
    Instrument          Digits  Point      1 pip
    ==================  ======  =========  ===========================
    EURUSD              5       0.00001    0.0001  (10 points)
    GBPUSD (legacy)     4       0.0001     0.0001  (1 point)
    USDJPY              3       0.001      0.01    (10 points)
    EURJPY (legacy)     2       0.01       0.01    (1 point)
    XAUUSD              2       0.01       0.1     (10 points, by convention)
    ==================  ======  =========  ===========================

    So: 10 points when the instrument carries a fractional pip digit (3
    or 5 decimals), otherwise 1 point — with metals and crypto pinned to
    10 points by convention.

    Returns
    -------
    float
        The price value of one pip, or ``0.0`` for a non-positive *point*.
    """
    if point <= 0:
        return 0.0
    if is_metal(symbol) or is_crypto(symbol):
        return point * 10
    return point * 10 if digits_from_point(point) in (3, 5) else point


def pips_to_price(pips: float, symbol: str, point: float) -> float:
    """Convert a pip distance into a price delta for *symbol*."""
    return pips * pip_size(symbol, point)


def price_to_pips(price_delta: float, symbol: str, point: float) -> float:
    """Convert a price delta into pips for *symbol*.

    The inverse of :func:`pips_to_price`.  Returns ``0.0`` when the pip
    size is unknown, so a bad ``point`` yields zero rather than a
    ``ZeroDivisionError`` mid-trade.
    """
    size = pip_size(symbol, point)
    if size == 0:
        return 0.0
    return price_delta / size


def estimate_pip_size(symbol: str) -> float:
    """Best-effort pip size for a caller that has no ``point``.

    Strategies receive only a symbol and an OHLCV frame — they never see
    ``symbol_info()`` — so a strategy that needs to reason in pips (the
    grid's level spacing, for example) has to assume a quote convention.
    This returns the pip size for each family's *standard* convention:

    ==========  ========
    Family      1 pip
    ==========  ========
    Metals      0.1
    Crypto      1.0
    JPY pairs   0.01
    Everything  0.0001
    ==========  ========

    Prefer :func:`pip_size` with the broker's real ``point`` wherever one
    is available — this is an estimate and will be wrong for a broker on
    a non-standard convention.

    The estimate exists here, next to the authoritative rule, because it
    previously lived inside ``GridStrategy`` as a two-line ``0.01`` /
    ``0.0001`` guess that read gold's pip as 0.0001 — a factor of 1000
    out, which mis-placed every grid level on metals.
    """
    if is_metal(symbol):
        return 0.1
    if is_crypto(symbol):
        return 1.0
    if "JPY" in _normalise(symbol):
        return 0.01
    return 0.0001


def pip_value_per_lot(
    symbol: str,
    point: float,
    *,
    contract_size: float = 0.0,
    tick_value: float = 0.0,
    tick_size: float = 0.0,
    fallback: float = 10.0,
) -> float:
    """Monetary value of a one-pip move on 1.0 lot, in the account currency.

    Three sources, in order of reliability:

    1. **Contract size**, for a symbol quoted in the account currency
       (``…USD``): the value is exactly ``contract_size × pip_size``.
       Preferred because some feeds report a bogus ``trade_tick_value``
       while the market is closed — MetaQuotes demo gold returns 0.1
       instead of 1.0 at weekends, which made every backtested gold trade
       lose ~$6.40 regardless of direction.
    2. **Broker tick data**, for a cross-quoted symbol (``EURJPY``…),
       where converting the quote currency needs the broker's own figure.
    3. **The fallback**, when neither is available.

    Parameters
    ----------
    symbol
        Instrument name.
    point
        The symbol's point size.
    contract_size
        ``symbol_info().trade_contract_size`` (0 if unknown).
    tick_value
        ``symbol_info().trade_tick_value`` (0 if unknown).
    tick_size
        ``symbol_info().trade_tick_size`` (0 if unknown).
    fallback
        Value to assume when the broker supplies nothing usable.

    Returns
    -------
    float
        Account-currency value of one pip per lot.
    """
    size = pip_size(symbol, point)
    if size <= 0:
        return fallback

    quoted_in_account_ccy = _normalise(symbol).endswith("USD")
    if quoted_in_account_ccy and contract_size > 0:
        return contract_size * size

    if tick_size > 0 and tick_value > 0:
        return tick_value * size / tick_size

    if contract_size > 0:
        return contract_size * size

    return fallback
