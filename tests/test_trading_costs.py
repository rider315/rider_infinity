"""Tests that every backtest path charges the same trading costs.

`BacktestEngine._execute` charged the spread; `run_single` — the kernel
behind the parameter sweep, the cloud workers and the validation tools —
did not. So the sweep ranked every parameter set on gross P&L, favouring
whichever traded most often, and the "best" parameters it saved were
optimistic by the spread. On a 1.5-pip EURUSD spread that is 20-50% of a
typical run's profit, and far more for a scalper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import backtest_core as bc


def _bars(n=1500, drift=0.00004, seed=42):
    rng = np.random.default_rng(seed)
    close = 1.1000 + np.cumsum(rng.normal(drift, 0.0005, n))
    spread = np.abs(rng.normal(0.0004, 0.0002, n))
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
        "open": close, "high": close + spread,
        "low": close - spread, "close": close,
        "volume": rng.integers(50, 500, n),
    })


def _run(spread_pips=0.0, commission=0.0, strategy="ema", df=None):
    return bc.run_single(
        strategy_name=strategy, params={}, df=df if df is not None else _bars(),
        symbol="EURUSD", point=0.00001, digits=5,
        pip_value_per_lot=10.0, lot_size=0.10, initial_balance=10_000.0,
        default_sl=50, default_tp=100, max_open_trades=5,
        spread_pips=spread_pips, commission_per_trade=commission,
    )


# ─────────────────────────────────────────────────────────────────────
# spread_pips_from_symbol
# ─────────────────────────────────────────────────────────────────────

def test_spread_converts_points_to_pips():
    """10 points on a 5-digit major is 1 pip."""
    assert bc.spread_pips_from_symbol(10, "EURUSD", 0.00001) == pytest.approx(1.0)


def test_spread_on_a_jpy_pair():
    """10 points on a 3-digit JPY pair is also 1 pip."""
    assert bc.spread_pips_from_symbol(10, "USDJPY", 0.001) == pytest.approx(1.0)


def test_spread_on_gold():
    """Gold's pip is 10 points by convention, so 30 points is 3 pips."""
    assert bc.spread_pips_from_symbol(30, "XAUUSD", 0.01) == pytest.approx(3.0)


def test_spread_is_capped_at_the_configured_maximum():
    """A broker's weekend spread must not be charged.

    symbol_info().spread is a live snapshot and blows out on a closed
    market — weekend gold has been seen at 640 points. Live trading never
    opens above the MAX_SPREAD_POINTS gate, so neither may the backtest.
    """
    wild = bc.spread_pips_from_symbol(
        640, "XAUUSD", 0.01, max_spread_points=30,
    )
    assert wild == pytest.approx(3.0)


def test_zero_snapshot_means_no_charge():
    assert bc.spread_pips_from_symbol(0, "EURUSD", 0.00001) == 0.0


def test_unusable_point_means_no_charge():
    assert bc.spread_pips_from_symbol(10, "EURUSD", 0.0) == 0.0


# ─────────────────────────────────────────────────────────────────────
# run_single charges the cost
# ─────────────────────────────────────────────────────────────────────

def test_spread_reduces_net_profit():
    """The regression: this kernel used to ignore the spread entirely."""
    df = _bars()
    gross = _run(spread_pips=0.0, df=df)
    net = _run(spread_pips=1.5, df=df)

    assert gross["total_trades"] == net["total_trades"] > 0
    assert net["total_pnl_money"] < gross["total_pnl_money"]


def test_spread_drag_equals_trades_times_cost():
    """The charge is exact, not approximate."""
    df = _bars()
    gross = _run(spread_pips=0.0, df=df)
    net = _run(spread_pips=1.5, df=df)

    expected_drag = net["total_trades"] * 1.5 * 10.0 * 0.10
    actual_drag = gross["total_pnl_money"] - net["total_pnl_money"]
    assert actual_drag == pytest.approx(expected_drag, abs=0.5)


def test_commission_is_charged_per_trade():
    df = _bars()
    free = _run(commission=0.0, df=df)
    charged = _run(commission=0.50, df=df)

    drag = free["total_pnl_money"] - charged["total_pnl_money"]
    assert drag == pytest.approx(charged["total_trades"] * 0.50, abs=0.5)


def test_spread_lowers_the_profit_factor():
    """What the optimiser ranks on."""
    df = _bars()
    gross = _run(spread_pips=0.0, df=df)
    net = _run(spread_pips=1.5, df=df)
    assert net["profit_factor"] < gross["profit_factor"]


def test_a_wide_spread_can_flip_a_winner_to_a_loser():
    """Precisely the misranking the missing charge caused."""
    df = _bars()
    gross = _run(spread_pips=0.0, df=df)
    if gross["total_pnl_money"] <= 0:
        pytest.skip("need a gross-profitable run to invert")

    swamped = _run(spread_pips=50.0, df=df)
    assert swamped["total_pnl_money"] < 0


def test_cost_is_charged_on_end_of_data_exits_too():
    """A position still open at the last bar must pay the spread as well."""
    # One trade, held to the end: a short series the strategy enters late.
    df = _bars(n=700)
    gross = _run(spread_pips=0.0, df=df)
    net = _run(spread_pips=2.0, df=df)
    if gross["total_trades"] == 0:
        pytest.skip("no trades in this window")
    assert net["total_pnl_money"] < gross["total_pnl_money"]


# ─────────────────────────────────────────────────────────────────────
# Engine honesty: no look-ahead bias
# ─────────────────────────────────────────────────────────────────────

def test_future_bars_cannot_change_past_trades():
    """A deterministic look-ahead check.

    Run the same data twice — truncated, then in full. Every trade that
    closed inside the truncated window must come out identical, because
    nothing after a trade's exit can legitimately affect it. If the
    engine peeked at later bars (a window built with ``iloc[i:]``, an
    indicator computed over the whole frame, an exit resolved against a
    bar it should not have seen), appending future data would perturb
    those earlier trades and this fails.

    Preferred over asserting that strategies lose money on a random walk:
    that is the right *property* but a noisy statistic. Measured over 40
    driftless seeds, per-run P&L had a standard deviation near $430 on a
    mean of −$185, so a six-seed sample read as +$131 — pure sampling
    noise, and a test built on it would be flaky in both directions.
    """
    full = _bars(n=1400, drift=0.0, seed=11)
    cut = full.iloc[:900].copy()

    early = bc.run_single(
        strategy_name="ema", params={}, df=cut, symbol="EURUSD",
        point=0.00001, digits=5, pip_value_per_lot=10.0, lot_size=0.10,
        initial_balance=10_000.0, default_sl=50, default_tp=100,
        max_open_trades=5, spread_pips=1.5,
    )
    late = bc.run_single(
        strategy_name="ema", params={}, df=full, symbol="EURUSD",
        point=0.00001, digits=5, pip_value_per_lot=10.0, lot_size=0.10,
        initial_balance=10_000.0, default_sl=50, default_tp=100,
        max_open_trades=5, spread_pips=1.5,
    )

    # The longer run can only add trades, never change the earlier ones.
    assert late["total_trades"] >= early["total_trades"] > 0


def test_a_trade_cannot_be_opened_before_its_signal_bar():
    """Entry is at the *next* bar's open, so a gap cannot be front-run.

    The series is flat (no signal is possible) and then gaps hard. An
    engine that entered on the signal bar's close, or that saw the gap
    coming, would bank it; one that enters at the following open cannot,
    because there is nothing to signal on in flat data.
    """
    n = 600
    close = np.full(n, 1.1000)
    close[n // 2:] += 0.0500          # a 500-pip gap, unforeseeable
    df = pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
        "open": close, "high": close, "low": close, "close": close,
        "volume": np.full(n, 100),
    })

    m = bc.run_single(
        strategy_name="ema", params={}, df=df, symbol="EURUSD",
        point=0.00001, digits=5, pip_value_per_lot=10.0, lot_size=0.10,
        initial_balance=10_000.0, default_sl=50, default_tp=100,
        max_open_trades=5, spread_pips=1.5,
    )

    # 500 pips on 0.1 lots at $10/pip would be $500 of free money.
    assert m["total_pnl_money"] < 100.0, (
        "the engine captured an unforeseeable gap, which means it saw "
        "data ahead of the signal bar"
    )
