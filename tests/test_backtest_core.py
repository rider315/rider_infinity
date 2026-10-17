"""Tests for the shared simulation kernel.

Every backtest path (in-process engine, parameter sweep, cloud worker,
A/B script) resolves exits and computes statistics through here, so these
numbers decide which strategies look profitable.
"""

from __future__ import annotations

import math

import pytest

import backtest_core as bc


# ─────────────────────────────────────────────────────────────────────
# check_sl_tp
# ─────────────────────────────────────────────────────────────────────

def _buy(sl=1.0900, tp=1.1100):
    return {"direction": "BUY", "sl_price": sl, "tp_price": tp}


def _sell(sl=1.1100, tp=1.0900):
    return {"direction": "SELL", "sl_price": sl, "tp_price": tp}


def _bar(high, low):
    return {"high": high, "low": low}


def test_buy_take_profit_hit():
    assert bc.check_sl_tp(_buy(), _bar(1.1150, 1.1000)) == (1.1100, "TP")


def test_buy_stop_loss_hit():
    assert bc.check_sl_tp(_buy(), _bar(1.1000, 1.0850)) == (1.0900, "SL")


def test_buy_untouched_bar():
    assert bc.check_sl_tp(_buy(), _bar(1.1050, 1.0950)) is None


def test_sell_take_profit_hit():
    assert bc.check_sl_tp(_sell(), _bar(1.1000, 1.0850)) == (1.0900, "TP")


def test_sell_stop_loss_hit():
    assert bc.check_sl_tp(_sell(), _bar(1.1150, 1.1000)) == (1.1100, "SL")


@pytest.mark.parametrize("trade", [_buy(), _sell()])
def test_both_levels_in_one_bar_assumes_the_stop(trade):
    """Intrabar order is unknowable, so the pessimistic result is correct.

    Assuming the target instead would inflate every backtest, and the
    error compounds over thousands of trades.
    """
    exit_price, reason = bc.check_sl_tp(trade, _bar(1.1200, 1.0800))
    assert reason == "SL"
    assert exit_price == trade["sl_price"]


def test_exact_touch_counts_as_a_hit():
    """Price reaching the level exactly fills — brokers do not round away."""
    assert bc.check_sl_tp(_buy(), _bar(1.1100, 1.1000))[1] == "TP"
    assert bc.check_sl_tp(_buy(), _bar(1.1000, 1.0900))[1] == "SL"


# ─────────────────────────────────────────────────────────────────────
# P&L
# ─────────────────────────────────────────────────────────────────────

def test_buy_pnl_pips_sign():
    assert bc.calc_pnl_pips(
        "BUY", 1.1000, 1.1020, "EURUSD", 0.00001,
    ) == pytest.approx(20.0)
    assert bc.calc_pnl_pips(
        "BUY", 1.1000, 1.0980, "EURUSD", 0.00001,
    ) == pytest.approx(-20.0)


def test_sell_pnl_pips_sign():
    """A SELL profits when price falls."""
    assert bc.calc_pnl_pips(
        "SELL", 1.1000, 1.0980, "EURUSD", 0.00001,
    ) == pytest.approx(20.0)
    assert bc.calc_pnl_pips(
        "SELL", 1.1000, 1.1020, "EURUSD", 0.00001,
    ) == pytest.approx(-20.0)


def test_jpy_pnl_uses_the_jpy_pip():
    """A 0.20 move on a 3-digit JPY pair is 20 pips, not 2."""
    assert bc.calc_pnl_pips(
        "BUY", 150.00, 150.20, "USDJPY", 0.001,
    ) == pytest.approx(20.0)


def test_pnl_money_charges_spread_and_commission():
    # 20 pips, 0.10 lots, $10/pip/lot, 1 pip spread, $0.50 commission
    assert bc.calc_pnl_money(
        20.0, 0.10, 10.0, spread_pips=1.0, commission=0.50,
    ) == pytest.approx((20 - 1) * 0.10 * 10.0 - 0.50)


def test_spread_can_turn_a_marginal_winner_into_a_loser():
    assert bc.calc_pnl_money(0.5, 1.0, 10.0, spread_pips=2.0) < 0


# ─────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────

def test_profit_factor_basic():
    assert bc.profit_factor([30.0, -10.0]) == pytest.approx(3.0)


def test_profit_factor_is_capped_not_infinite():
    """float('inf') is not JSON-serialisable and broke the API payload."""
    pf = bc.profit_factor([10.0, 20.0])
    assert pf == 999.99
    assert math.isfinite(pf)


def test_profit_factor_with_no_wins_is_zero():
    assert bc.profit_factor([-10.0, -5.0]) == 0.0


def test_profit_factor_of_nothing_is_zero():
    assert bc.profit_factor([]) == 0.0


def test_profit_factor_ignores_breakeven_trades():
    assert bc.profit_factor([30.0, -10.0, 0.0]) == pytest.approx(3.0)


def test_sharpe_needs_two_trades():
    assert bc.sharpe_ratio([]) == 0.0
    assert bc.sharpe_ratio([5.0]) == 0.0


def test_sharpe_of_constant_returns_is_zero():
    """Zero variance must not divide by zero."""
    assert bc.sharpe_ratio([5.0, 5.0, 5.0]) == 0.0


def test_sharpe_is_positive_for_a_profitable_series():
    assert bc.sharpe_ratio([10.0, 12.0, 8.0, 11.0]) > 0


def test_sharpe_is_negative_for_a_losing_series():
    assert bc.sharpe_ratio([-10.0, -12.0, -8.0, -11.0]) < 0


def test_sharpe_uses_the_sample_standard_deviation():
    pnls = [10.0, 20.0]
    mean = 15.0
    sample_sd = math.sqrt(((10 - mean) ** 2 + (20 - mean) ** 2) / 1)
    assert bc.sharpe_ratio(pnls) == pytest.approx(
        mean / sample_sd * bc.SQRT_TRADING_DAYS
    )


# ── Equity curve / drawdown ──────────────────────────────────────────

def test_equity_curve_starts_at_the_balance():
    assert bc.equity_curve_from_trades([10, -5], 100) == [100, 110, 105]


def test_max_drawdown_measures_from_the_running_peak():
    money, pct = bc.max_drawdown([100, 120, 90, 110])
    assert money == pytest.approx(30.0)
    assert pct == pytest.approx(25.0)


def test_max_drawdown_of_a_rising_curve_is_zero():
    assert bc.max_drawdown([100, 110, 120]) == (0.0, 0.0)


def test_max_drawdown_of_empty_curve():
    assert bc.max_drawdown([]) == (0.0, 0.0)


def test_max_drawdown_keeps_the_worst_not_the_last():
    _, pct = bc.max_drawdown([100, 50, 100, 90])
    assert pct == pytest.approx(50.0)


# ── Consecutive streaks ──────────────────────────────────────────────

def test_max_consecutive_wins():
    assert bc.max_consecutive([True, True, False, True, True, True]) == 3


def test_max_consecutive_losses():
    assert bc.max_consecutive(
        [True, False, False, True], target=False,
    ) == 2


def test_max_consecutive_of_nothing():
    assert bc.max_consecutive([]) == 0
