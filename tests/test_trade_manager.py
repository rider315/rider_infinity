"""Tests for the shared trade-management rules.

These rules decide how much a trade may lose and when it is cut, so they
are the highest-value thing in the code-base to pin down.  Both engines
(``multi_bot.py`` and ``bot.py``) route through here, so a regression
would hit live trading on every path at once.
"""

from __future__ import annotations

import pandas as pd
import pytest

import trade_manager as tm
from strategies.base import Signal, SignalAction

EURUSD_POINT = 0.00001   # 5-digit major: 1 pip = 10 points
JPY_POINT = 0.001        # 3-digit JPY pair: 1 pip = 100 points
BTC_POINT = 0.01


def _frame(closes, high_offset=0.0005, low_offset=0.0005):
    """Build a minimal OHLC frame with a controllable true range."""
    return pd.DataFrame({
        "open": closes,
        "high": [c + high_offset for c in closes],
        "low": [c - low_offset for c in closes],
        "close": closes,
        "volume": [100] * len(closes),
    })


def _signal(sl=0.0, tp=0.0):
    return Signal(action=SignalAction.BUY, symbol="EURUSD",
                  sl_pips=sl, tp_pips=tp)


# ─────────────────────────────────────────────────────────────────────
# resolve_sl_tp
# ─────────────────────────────────────────────────────────────────────

def test_fixed_pips_pass_through_when_atr_off(config):
    config.use_atr_stops = False
    sl, tp = tm.resolve_sl_tp(
        config, _signal(sl=20, tp=60), _frame([1.1] * 50),
        "EURUSD", EURUSD_POINT,
    )
    assert sl == 20
    assert tp == 60


def test_strategy_default_is_clamped_to_max(config):
    """A strategy's wide default stop cannot exceed max_sl_pips."""
    config.use_atr_stops = False
    config.max_sl_pips = 60
    sl, _ = tm.resolve_sl_tp(
        config, _signal(sl=500, tp=1000), _frame([1.1] * 50),
        "EURUSD", EURUSD_POINT,
    )
    assert sl == 60


def test_tiny_stop_is_raised_to_min(config):
    config.use_atr_stops = False
    config.min_sl_pips = 8
    sl, _ = tm.resolve_sl_tp(
        config, _signal(sl=0.5, tp=1), _frame([1.1] * 50),
        "EURUSD", EURUSD_POINT,
    )
    assert sl == 8


def test_rr_floor_is_enforced(config):
    """A 5-pip target against a 20-pip stop needs a >80% win-rate."""
    config.use_atr_stops = False
    config.min_rr_ratio = 1.8
    sl, tp = tm.resolve_sl_tp(
        config, _signal(sl=20, tp=5), _frame([1.1] * 50),
        "EURUSD", EURUSD_POINT,
    )
    assert sl == 20
    assert tp == pytest.approx(36.0)


def test_generous_tp_is_not_reduced_by_the_rr_floor(config):
    config.use_atr_stops = False
    config.min_rr_ratio = 1.8
    _, tp = tm.resolve_sl_tp(
        config, _signal(sl=10, tp=100), _frame([1.1] * 50),
        "EURUSD", EURUSD_POINT,
    )
    assert tp == 100


def test_atr_stop_scales_with_volatility(config):
    """The whole point of ATR stops: a wilder instrument gets a wider stop."""
    config.use_atr_stops = True
    config.atr_sl_mult = 1.5
    config.min_sl_pips = 1
    config.max_sl_pips = 500

    calm, _ = tm.resolve_sl_tp(
        config, _signal(), _frame([1.1] * 50, 0.0002, 0.0002),
        "EURUSD", EURUSD_POINT,
    )
    wild, _ = tm.resolve_sl_tp(
        config, _signal(), _frame([1.1] * 50, 0.0020, 0.0020),
        "EURUSD", EURUSD_POINT,
    )
    assert wild > calm * 5


def test_atr_stop_tp_follows_the_rr_ratio(config):
    config.use_atr_stops = True
    config.min_rr_ratio = 2.0
    config.min_sl_pips = 1
    config.max_sl_pips = 500
    sl, tp = tm.resolve_sl_tp(
        config, _signal(), _frame([1.1] * 50), "EURUSD", EURUSD_POINT,
    )
    assert tp == pytest.approx(sl * 2.0, rel=1e-6)


def test_jpy_pip_size_is_honoured(config):
    """A 3-digit JPY pip is 0.01, so a 20-pip stop must be 0.20 away."""
    config.use_atr_stops = False
    sl, _ = tm.resolve_sl_tp(
        config, _signal(sl=20, tp=40), _frame([150.0] * 50),
        "USDJPY", JPY_POINT,
    )
    sl_price = tm.pips_to_price(sl, "USDJPY", JPY_POINT)
    assert sl_price == pytest.approx(0.20)


# ── Crypto: percentage-of-price band, not forex pip bounds ───────────

def test_crypto_stop_ignores_the_forex_pip_clamp(config):
    """A 60-pip stop on a $60k instrument is dust and stops out instantly."""
    config.use_atr_stops = False  # crypto forces ATR regardless
    config.max_sl_pips = 60
    sl, _ = tm.resolve_sl_tp(
        config, _signal(sl=50, tp=100), _frame([60_000.0] * 50, 50, 50),
        "BTCUSD", BTC_POINT,
    )
    assert sl > 60


def test_crypto_stop_stays_inside_the_percentage_band(config):
    price = 60_000.0
    sl, _ = tm.resolve_sl_tp(
        config, _signal(), _frame([price] * 50, 5000, 5000),
        "BTCUSD", BTC_POINT,
    )
    sl_price = tm.pips_to_price(sl, "BTCUSD", BTC_POINT)
    assert price * 0.0015 <= sl_price <= price * 0.015 * 1.01


def test_crypto_without_atr_uses_the_percentage_fallback(config):
    """Flat candles give ATR 0, so the fallback must still size the stop."""
    price = 50_000.0
    sl, _ = tm.resolve_sl_tp(
        config, _signal(), _frame([price] * 50, 0.0, 0.0),
        "ETHUSD", BTC_POINT,
    )
    sl_price = tm.pips_to_price(sl, "ETHUSD", BTC_POINT)
    assert sl_price == pytest.approx(price * 0.004, rel=0.01)


# ─────────────────────────────────────────────────────────────────────
# cap_sl_for_quick_exit
# ─────────────────────────────────────────────────────────────────────

def test_quick_exit_cap_fixes_inverted_risk_reward(config):
    """A 5-pip target against a 50-pip stop needs a >90% win-rate."""
    config.quick_exit_sl_cap_mult = 3.0
    config.min_sl_pips = 8
    sl, note = tm.cap_sl_for_quick_exit(config, sl_pips=50, quick_exit_pips=5)
    assert sl == 15.0
    assert "capped" in note


def test_quick_exit_cap_never_goes_below_min_sl(config):
    config.quick_exit_sl_cap_mult = 3.0
    config.min_sl_pips = 8
    sl, _ = tm.cap_sl_for_quick_exit(config, sl_pips=50, quick_exit_pips=1)
    assert sl == 8.0


def test_quick_exit_cap_leaves_a_proportionate_stop_alone(config):
    config.quick_exit_sl_cap_mult = 3.0
    sl, note = tm.cap_sl_for_quick_exit(config, sl_pips=10, quick_exit_pips=5)
    assert sl == 10
    assert note is None


def test_quick_exit_cap_disabled_by_zero(config):
    config.quick_exit_sl_cap_mult = 0
    sl, note = tm.cap_sl_for_quick_exit(config, sl_pips=50, quick_exit_pips=5)
    assert (sl, note) == (50, None)


# ─────────────────────────────────────────────────────────────────────
# Breakeven
# ─────────────────────────────────────────────────────────────────────

def _pos(**kw):
    base = dict(ticket=1, type="BUY", entry=1.1000, sl=1.0980,
                price_current=1.1000, open_time=0)
    base.update(kw)
    return base


def test_breakeven_fires_at_the_trigger_multiple(config):
    config.breakeven_enabled = True
    config.breakeven_trigger_r = 0.6
    # risk = 20 pips; 0.6R = 12 pips in profit
    out = tm.breakeven_decisions(config, [_pos(price_current=1.1013)])
    assert len(out) == 1
    assert out[0]["ticket"] == 1
    assert out[0]["r_multiple"] == pytest.approx(0.65, abs=0.01)


def test_breakeven_does_not_fire_too_early(config):
    config.breakeven_enabled = True
    config.breakeven_trigger_r = 0.6
    assert tm.breakeven_decisions(config, [_pos(price_current=1.1005)]) == []


def test_breakeven_skips_a_stop_already_at_entry(config):
    """Avoids spamming modify requests on an already-protected trade."""
    config.breakeven_enabled = True
    out = tm.breakeven_decisions(
        config, [_pos(sl=1.1000, price_current=1.1050)],
    )
    assert out == []


def test_breakeven_handles_sell_direction(config):
    config.breakeven_enabled = True
    config.breakeven_trigger_r = 0.6
    out = tm.breakeven_decisions(config, [_pos(
        type="SELL", entry=1.1000, sl=1.1020, price_current=1.0987,
    )])
    assert len(out) == 1


def test_breakeven_skips_sell_already_protected(config):
    config.breakeven_enabled = True
    out = tm.breakeven_decisions(config, [_pos(
        type="SELL", entry=1.1000, sl=1.1000, price_current=1.0900,
    )])
    assert out == []


def test_breakeven_respects_the_disable_switch(config):
    config.breakeven_enabled = False
    assert tm.breakeven_decisions(config, [_pos(price_current=1.1100)]) == []


def test_breakeven_ignores_a_position_with_no_stop(config):
    """Risk is undefined without a stop, so there is no R to measure."""
    config.breakeven_enabled = True
    assert tm.breakeven_decisions(config, [_pos(sl=0.0)]) == []


# ─────────────────────────────────────────────────────────────────────
# Time-stop
# ─────────────────────────────────────────────────────────────────────

def test_time_stop_closes_a_stale_position(config):
    config.max_trade_hours = 48
    now = 1_000_000.0
    out = tm.time_stop_decisions(
        config, [_pos(open_time=now - 49 * 3600)], now=now,
    )
    assert len(out) == 1
    assert out[0]["age_hours"] == pytest.approx(49.0)


def test_time_stop_leaves_a_fresh_position(config):
    config.max_trade_hours = 48
    now = 1_000_000.0
    out = tm.time_stop_decisions(
        config, [_pos(open_time=now - 2 * 3600)], now=now,
    )
    assert out == []


def test_time_stop_disabled_by_zero(config):
    config.max_trade_hours = 0
    now = 1_000_000.0
    out = tm.time_stop_decisions(
        config, [_pos(open_time=now - 9999 * 3600)], now=now,
    )
    assert out == []


# ─────────────────────────────────────────────────────────────────────
# Post-loss cooldown
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("reason,profit,expected", [
    ("SL", -12.0, True),        # the loss-churn case this exists for
    ("STOP_OUT", -50.0, True),
    ("SL", 4.0, False),         # trailed Quick-Exit stop = banked winner
    ("TP", 20.0, False),
    ("MANUAL", -12.0, False),   # operator chose to close; not a failed entry
])
def test_cooldown_only_arms_on_a_losing_stop_out(config, reason, profit, expected):
    config.sl_cooldown_min = 15
    assert tm.should_start_cooldown(config, reason, profit) is expected


def test_cooldown_disabled_by_zero(config):
    config.sl_cooldown_min = 0
    assert tm.should_start_cooldown(config, "SL", -12.0) is False


def test_cooldown_blocks_then_expires(config):
    config.sl_cooldown_min = 15
    now = 1_000_000.0
    until = tm.cooldown_expiry(config, now=now)
    assert until == now + 900

    assert tm.cooldown_block_reason(until, now=now) is not None
    assert "14m" in tm.cooldown_block_reason(until, now=now + 60)
    assert tm.cooldown_block_reason(until, now=now + 900) is None


# ─────────────────────────────────────────────────────────────────────
# Price levels
# ─────────────────────────────────────────────────────────────────────

def test_buy_stop_below_and_target_above():
    sl, tp = tm.stop_and_target_prices(
        "BUY", 1.10000, 20, 40, "EURUSD", EURUSD_POINT, 5,
    )
    assert sl == 1.09800
    assert tp == 1.10400


def test_sell_stop_above_and_target_below():
    sl, tp = tm.stop_and_target_prices(
        "SELL", 1.10000, 20, 40, "EURUSD", EURUSD_POINT, 5,
    )
    assert sl == 1.10200
    assert tp == 1.09600


def test_price_levels_round_to_symbol_digits():
    sl, tp = tm.stop_and_target_prices(
        "BUY", 150.123, 20, 40, "USDJPY", JPY_POINT, 3,
    )
    assert sl == round(sl, 3)
    assert tp == round(tp, 3)


# ─────────────────────────────────────────────────────────────────────
# position_view adapter
# ─────────────────────────────────────────────────────────────────────

def test_position_view_maps_an_mt5_record(mt5):
    raw = mt5.Position(
        ticket=42, type=mt5.ORDER_TYPE_SELL, price_open=1.2,
        price_current=1.1, sl=1.3, tp=1.0, time=12345,
    )
    view = tm.position_view(raw)
    assert view["ticket"] == 42
    assert view["type"] == "SELL"
    assert view["entry"] == 1.2
    assert view["open_time"] == 12345
    # The shape the rule functions consume.
    assert tm.breakeven_decisions.__name__  # sanity
    assert set(view) >= {
        "ticket", "type", "entry", "sl", "price_current", "open_time",
    }


# ─────────────────────────────────────────────────────────────────────
# Pip conventions (utils.pip_size)
#
# Every stop, target, journalled pip count and backtest P&L flows
# through this, so an error here silently mis-sizes every trade on the
# affected instrument.
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("symbol,point,expected_pip", [
    # 5-digit majors: fractional pip digit -> 10 points.
    ("EURUSD", 0.00001, 0.0001),
    ("GBPUSD", 0.00001, 0.0001),
    # Legacy 4-digit quote: no fractional pip -> 1 point.
    ("GBPUSD", 0.0001, 0.0001),
    # 3-digit JPY pairs. The old "100 * point if JPY" rule returned 0.1
    # here, so a 20-pip stop was placed 200 pips away.
    ("USDJPY", 0.001, 0.01),
    ("GBPJPY", 0.001, 0.01),
    ("EURJPY", 0.001, 0.01),
    # Legacy 2-digit JPY quote.
    ("EURJPY", 0.01, 0.01),
    # Metals: 2 digits but 0.1 is one pip by convention.
    ("XAUUSD", 0.01, 0.1),
    ("XAGUSD", 0.001, 0.01),
])
def test_pip_size_matches_quote_convention(symbol, point, expected_pip):
    assert tm.pip_size(symbol, point) == pytest.approx(expected_pip)


def test_pip_size_tolerates_broker_decorations():
    assert tm.pip_size("XAUUSD.pro", 0.01) == pytest.approx(0.1)
    assert tm.pip_size("GBPJPYm", 0.001) == pytest.approx(0.01)


def test_pip_size_of_unknown_point_is_zero():
    assert tm.pip_size("EURUSD", 0.0) == 0.0
    assert tm.price_to_pips(1.0, "EURUSD", 0.0) == 0.0


@pytest.mark.parametrize("symbol,point", [
    ("EURUSD", 0.00001), ("GBPJPY", 0.001), ("XAUUSD", 0.01),
    ("BTCUSD", 0.01), ("GBPUSD", 0.0001),
])
def test_pip_conversion_round_trips(symbol, point):
    """pips -> price -> pips must be the identity."""
    assert tm.price_to_pips(
        tm.pips_to_price(37.5, symbol, point), symbol, point,
    ) == pytest.approx(37.5)


def test_jpy_stop_is_sized_in_real_pips(config):
    """The regression: a configured 20-pip stop must be 20 real pips."""
    config.use_atr_stops = False
    sl, _ = tm.resolve_sl_tp(
        config, _signal(sl=20, tp=40), _frame([150.0] * 50),
        "GBPJPY", JPY_POINT,
    )
    assert tm.pips_to_price(sl, "GBPJPY", JPY_POINT) == pytest.approx(0.20)


def test_major_pip_size_is_unchanged_by_the_fix():
    """Non-JPY instruments must keep their previous pip size exactly."""
    assert tm.pips_to_price(20, "EURUSD", 0.00001) == pytest.approx(0.0020)
    assert tm.pips_to_price(20, "XAUUSD", 0.01) == pytest.approx(2.0)
