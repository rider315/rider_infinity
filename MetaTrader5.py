"""
Stub module for MetaTrader5 on Linux.
MT5 only runs on Windows - this prevents import errors on the dashboard.
All functions return realistic mock data so the dashboard and bots can run.
"""
import time
import numpy as np

# ── Timeframe constants ──
TIMEFRAME_M1 = 1
TIMEFRAME_M5 = 5
TIMEFRAME_M15 = 15
TIMEFRAME_M30 = 30
TIMEFRAME_H1 = 16385
TIMEFRAME_H4 = 16388
TIMEFRAME_D1 = 16408
TIMEFRAME_W1 = 32769
TIMEFRAME_MN1 = 49153

# ── Trade action constants ──
TRADE_ACTION_DEAL = 1
TRADE_ACTION_PENDING = 5
TRADE_ACTION_SLTP = 6
TRADE_ACTION_MODIFY = 7
TRADE_ACTION_REMOVE = 8

# ── Order type constants ──
ORDER_TYPE_BUY = 0
ORDER_TYPE_SELL = 1
ORDER_TYPE_BUY_LIMIT = 2
ORDER_TYPE_SELL_LIMIT = 3
ORDER_TYPE_BUY_STOP = 4
ORDER_TYPE_SELL_STOP = 5
ORDER_TYPE_BUY_STOP_LIMIT = 6
ORDER_TYPE_SELL_STOP_LIMIT = 7

# ── Order filling constants ──
ORDER_FILLING_FOK = 0
ORDER_FILLING_IOC = 1
ORDER_FILLING_RETURN = 2

# ── Order time constants ──
ORDER_TIME_GTC = 0
ORDER_TIME_DAY = 1
ORDER_TIME_SPECIFIED = 2

# ── Trade return codes ──
TRADE_RETCODE_DONE = 10009
TRADE_RETCODE_PLACED = 10008

# ── Position type constants ──
POSITION_TYPE_BUY = 0
POSITION_TYPE_SELL = 1

# ── Symbol trade mode ──
SYMBOL_TRADE_MODE_DISABLED = 0


# ── Base prices for different symbols ──
_BASE_PRICES = {
    "EURUSD": 1.1000, "GBPUSD": 1.2700, "USDJPY": 150.00,
    "GBPJPY": 190.50, "EURJPY": 165.00, "AUDUSD": 0.6600,
    "USDCHF": 0.8800, "NZDUSD": 0.6100, "USDCAD": 1.3600,
    "EURGBP": 0.8650, "XAUUSD": 2350.00, "XAGUSD": 29.50,
}


class _MockObj:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
    def _asdict(self):
        return self.__dict__.copy()


def _get_base(symbol):
    return _BASE_PRICES.get(symbol, 1.1000)


def _mock_account():
    return _MockObj(
        login=0, name="Demo (Linux Stub)", server="StubServer",
        balance=0.0, equity=0.0, margin=0.0, margin_free=0.0,
        margin_level=0.0, profit=0.0, currency="USD", leverage=100,
        trade_mode=0, limit_orders=200, company="Stub Broker",
    )


def _mock_symbol(symbol="EURUSD"):
    base = _get_base(symbol)
    is_jpy = "JPY" in symbol
    point = 0.001 if is_jpy else 0.00001
    digits = 3 if is_jpy else 5
    if symbol.startswith("XAU"):
        point, digits = 0.01, 2
    elif symbol.startswith("XAG"):
        point, digits = 0.001, 3
    return _MockObj(
        name=symbol, visible=True, trade_mode=1, point=point, digits=digits,
        spread=10, volume_min=0.01, volume_max=100.0, volume_step=0.01,
        trade_contract_size=100000, trade_tick_size=point, trade_tick_value=1.0,
        ask=base + 10 * point, bid=base,
    )


def _mock_tick(symbol="EURUSD"):
    base = _get_base(symbol)
    is_jpy = "JPY" in symbol
    point = 0.001 if is_jpy else 0.00001
    if symbol.startswith("XAU"):
        point = 0.01
    elif symbol.startswith("XAG"):
        point = 0.001
    return _MockObj(
        ask=base + 10 * point, bid=base, last=base + 5 * point,
        volume=100, time=int(time.time()),
    )


def initialize(*args, **kwargs): return True
def shutdown(*args, **kwargs): pass
def login(*args, **kwargs): return True
def terminal_info(*args, **kwargs): return _MockObj(name="Stub Terminal", connected=True)
def account_info(*args, **kwargs): return _mock_account()
def symbol_info(symbol=None, *args, **kwargs): return _mock_symbol(symbol or "EURUSD")
def symbol_info_tick(symbol=None, *args, **kwargs): return _mock_tick(symbol or "EURUSD")
def symbol_select(*args, **kwargs): return True


def copy_rates_from_pos(symbol, timeframe, start_pos, count):
    """Generate fake OHLCV candle data as a numpy structured array."""
    base = _get_base(symbol if symbol else "EURUSD")
    is_jpy = "JPY" in (symbol or "")
    noise_scale = 0.5 if is_jpy else (5.0 if "XAU" in (symbol or "") else 0.001)

    now = int(time.time())
    # Map timeframe to seconds
    tf_secs = {1: 60, 5: 300, 15: 900, 30: 1800,
               16385: 3600, 16388: 14400, 16408: 86400,
               32769: 604800, 49153: 2592000}
    interval = tf_secs.get(timeframe, 300)

    dtype = np.dtype([
        ('time', 'i8'), ('open', 'f8'), ('high', 'f8'),
        ('low', 'f8'), ('close', 'f8'), ('tick_volume', 'i8'),
        ('spread', 'i4'), ('real_volume', 'i8'),
    ])
    rates = np.zeros(count, dtype=dtype)

    price = base
    for i in range(count):
        t = now - (count - i) * interval
        change = np.random.normal(0, noise_scale)
        o = price
        c = o + change
        h = max(o, c) + abs(np.random.normal(0, noise_scale * 0.5))
        l = min(o, c) - abs(np.random.normal(0, noise_scale * 0.5))
        rates[i] = (t, round(o, 5), round(h, 5), round(l, 5),
                     round(c, 5), np.random.randint(50, 500), 10, 0)
        price = c

    return rates


def copy_rates_range(*args, **kwargs): return copy_rates_from_pos("EURUSD", 5, 0, 200)
def copy_ticks_from(*args, **kwargs): return None
def positions_get(*args, **kwargs): return ()
def positions_total(*args, **kwargs): return 0
def orders_get(*args, **kwargs): return ()
def orders_total(*args, **kwargs): return 0
def history_orders_get(*args, **kwargs): return ()
def history_deals_get(*args, **kwargs): return ()
def order_send(*args, **kwargs): return _MockObj(retcode=TRADE_RETCODE_DONE, order=0, comment="Stub - no real execution")
def order_check(*args, **kwargs): return _MockObj(retcode=TRADE_RETCODE_DONE)
def last_error(*args, **kwargs): return (0, "Stub - MT5 not available on Linux")
