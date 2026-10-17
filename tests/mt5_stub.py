r"""In-process stand-in for the ``MetaTrader5`` package.

The real package is Windows-only, so every module that does
``import MetaTrader5 as mt5`` is unimportable on Linux/CI.  This stub
provides the constants and the function surface the code-base actually
touches (see ``grep -rhoE 'mt5\.[A-Za-z_]+'``) so the pure logic can be
unit-tested without a terminal.

Functions default to "nothing there" (``None`` / empty tuple).  Tests
override individual attributes, e.g.::

    mt5.symbol_info = lambda s: SymbolInfo(point=0.00001, digits=5)

``reset()`` restores the defaults between tests.
"""

from __future__ import annotations

from types import SimpleNamespace

# ── Order types ──────────────────────────────────────────────────────
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TYPE_BUY_LIMIT = 2
ORDER_TYPE_SELL_LIMIT = 3
ORDER_TYPE_BUY_STOP = 4
ORDER_TYPE_SELL_STOP = 5
ORDER_TYPE_BUY_STOP_LIMIT = 6
ORDER_TYPE_SELL_STOP_LIMIT = 7

# ── Trade actions ────────────────────────────────────────────────────
TRADE_ACTION_DEAL = 1
TRADE_ACTION_PENDING = 5
TRADE_ACTION_SLTP = 6
TRADE_ACTION_MODIFY = 7
TRADE_ACTION_REMOVE = 8

# ── Retcodes ─────────────────────────────────────────────────────────
TRADE_RETCODE_PLACED = 10008
TRADE_RETCODE_DONE = 10009

# ── Filling / expiry ─────────────────────────────────────────────────
ORDER_FILLING_FOK = 0
ORDER_FILLING_IOC = 1
ORDER_FILLING_RETURN = 2
ORDER_TIME_GTC = 0
ORDER_TIME_SPECIFIED = 2

# ── Symbol trade modes ───────────────────────────────────────────────
SYMBOL_TRADE_MODE_DISABLED = 0

# ── Timeframes (real MT5 integer values) ─────────────────────────────
TIMEFRAME_M1 = 1
TIMEFRAME_M5 = 5
TIMEFRAME_M15 = 15
TIMEFRAME_M30 = 30
TIMEFRAME_H1 = 16385
TIMEFRAME_H4 = 16388
TIMEFRAME_D1 = 16408
TIMEFRAME_W1 = 32769


# ── Lightweight record factories ─────────────────────────────────────
def SymbolInfo(**kw):
    """A ``symbol_info()`` record with sane forex defaults."""
    base = dict(
        name="EURUSD", point=0.00001, digits=5, spread=10,
        volume_min=0.01, volume_max=100.0, volume_step=0.01,
        filling_mode=1, visible=True, trade_mode=4,
        trade_stops_level=0, trade_freeze_level=0,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def Tick(**kw):
    """A ``symbol_info_tick()`` record."""
    base = dict(bid=1.10000, ask=1.10010, last=1.10005, volume=0, time=0)
    base.update(kw)
    return SimpleNamespace(**base)


def Position(**kw):
    """A ``positions_get()`` record."""
    base = dict(
        ticket=1, symbol="EURUSD", type=ORDER_TYPE_BUY, volume=0.01,
        price_open=1.10000, price_current=1.10000, sl=0.0, tp=0.0,
        profit=0.0, swap=0.0, commission=0.0, magic=0, comment="",
        time=0, identifier=1,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def OrderResult(**kw):
    """An ``order_send()`` result (supports ``_asdict``)."""
    base = dict(
        retcode=TRADE_RETCODE_DONE, deal=0, order=0, volume=0.0,
        price=0.0, bid=0.0, ask=0.0, comment="", request_id=0,
        retcode_external=0,
    )
    base.update(kw)
    ns = SimpleNamespace(**base)
    ns._asdict = lambda: dict(base)
    return ns


def TerminalInfo(**kw):
    """A ``terminal_info()`` record."""
    base = dict(
        connected=True, trade_allowed=True, trade_expert=True,
        path=r"C:\MT5", data_path=r"C:\Users\t\AppData\MT5",
        name="MetaTrader 5", build=4000,
    )
    base.update(kw)
    return SimpleNamespace(**base)


def AccountInfo(**kw):
    """An ``account_info()`` record."""
    base = dict(
        login=1, name="demo", server="Broker-Demo", balance=10_000.0,
        equity=10_000.0, margin=0.0, margin_free=10_000.0,
        leverage=100, currency="USD",
    )
    base.update(kw)
    return SimpleNamespace(**base)


# ── Default (inert) API surface ──────────────────────────────────────
_DEFAULTS: dict = {}


def initialize(*a, **kw):
    return True


def shutdown():
    return None


def last_error():
    return (0, "stub: no error")


def terminal_info():
    return TerminalInfo()


def account_info():
    return AccountInfo()


def symbol_info(symbol):
    return SymbolInfo(name=symbol)


def symbol_info_tick(symbol):
    return Tick()


def symbol_select(symbol, enable=True):
    return True


def symbols_get(*a, **kw):
    return (SymbolInfo(name="EURUSD"),)


def positions_get(**kw):
    return ()


def orders_get(**kw):
    return ()


def order_send(request):
    return OrderResult()


def copy_rates_from_pos(symbol, timeframe, start, count):
    return None


def copy_rates_range(symbol, timeframe, date_from, date_to):
    return None


def history_deals_get(*a, **kw):
    return ()


_DEFAULTS = {
    name: obj for name, obj in list(globals().items())
    if callable(obj) and not name.startswith("_")
    and name[0].islower()
}


def reset() -> None:
    """Restore every stubbed function to its default implementation."""
    globals().update(_DEFAULTS)
