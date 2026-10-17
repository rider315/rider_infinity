"""Tests for strategy defaults that measurably change results.

A strategy's constructor defaults are what the fleet actually trades:
the dashboard creates bots with no strategy_params at all, so whatever
is in the signature is what goes live. A filter that exists but defaults
off is dead code with a live cost.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimizer import PARAM_GRIDS, generate_combos
from strategies import get_strategy
from strategies.ema_crossover import EMACrossoverStrategy


def _trending_bars(n=1200, seed=3):
    """Bars with a long uptrend and counter-trend pullbacks.

    The pullbacks are what a raw crossover trades and a trend filter
    rejects, so the two configurations must behave differently here.
    """
    rng = np.random.default_rng(seed)
    trend = np.linspace(0, 0.08, n)
    wobble = np.cumsum(rng.normal(0, 0.0012, n))
    close = 1.1000 + trend + wobble
    spread = np.abs(rng.normal(0.0004, 0.0002, n))
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC"),
        "open": close, "high": close + spread,
        "low": close - spread, "close": close,
        "volume": rng.integers(50, 500, n),
    })


# ─────────────────────────────────────────────────────────────────────
# EMA trend filter
# ─────────────────────────────────────────────────────────────────────

def test_trend_filter_is_on_by_default():
    """The regression.

    It shipped False, so the live default was the configuration this
    strategy's own comments called unprofitable. Measured over real daily
    S&P 500 and NASDAQ data (1999-2018), enabling it improved net P&L,
    profit factor and win rate in all six instrument x stop-distance
    combinations tested.
    """
    assert EMACrossoverStrategy().use_trend_filter is True
    assert get_strategy("ema").use_trend_filter is True


def test_trend_filter_can_still_be_turned_off():
    """The old behaviour stays reachable for comparison."""
    assert EMACrossoverStrategy(use_trend_filter=False).use_trend_filter is False


def test_the_filter_actually_rejects_counter_trend_crosses():
    """Its mechanism, not just its flag.

    The filter only takes crosses agreeing with the 200-period EMA, so it
    must produce strictly fewer signals than the raw crossover.
    """
    df = _trending_bars()
    raw = EMACrossoverStrategy(use_trend_filter=False)
    filtered = EMACrossoverStrategy(use_trend_filter=True)

    raw_signals = filtered_signals = 0
    for end in range(250, len(df), 5):
        window = df.iloc[:end]
        if raw.analyze(window, "EURUSD").action.value != "HOLD":
            raw_signals += 1
        if filtered.analyze(window, "EURUSD").action.value != "HOLD":
            filtered_signals += 1

    assert raw_signals > 0, "no crosses in this fixture"
    assert filtered_signals < raw_signals


def test_min_confidence_stays_off_by_default():
    """Deliberately unchanged.

    On the same real data it helped at some stop distances and hurt at
    others, and cut the sample to ~26 trades over 20 years — too thin to
    justify a default.
    """
    assert EMACrossoverStrategy().min_confidence == 0.0


# ─────────────────────────────────────────────────────────────────────
# The optimiser must be able to find it
# ─────────────────────────────────────────────────────────────────────

def test_the_sweep_searches_the_trend_filter():
    """It previously searched only the two periods.

    The highest-impact setting on the strategy was outside the grid, so
    no sweep — however deep — could have discovered it.
    """
    assert "use_trend_filter" in PARAM_GRIDS["ema"]


@pytest.mark.parametrize("mode", ["normal", "deep"])
def test_the_sweep_compares_both_settings(mode):
    combos = generate_combos("ema", mode, optimize_sltp=False)
    settings = {c["use_trend_filter"] for c in combos}
    assert settings == {True, False}


def test_quick_mode_only_tries_the_better_default():
    """Quick mode is for a fast answer, so it does not spend half the
    grid re-testing the configuration we already measured as worse."""
    combos = generate_combos("ema", "quick", optimize_sltp=False)
    assert {c["use_trend_filter"] for c in combos} == {True}


def test_every_graded_param_is_constructible():
    """A grid entry naming a parameter the constructor rejects would
    fail every combo in the sweep at runtime."""
    for strategy_name, grid in PARAM_GRIDS.items():
        params = {
            name: modes["normal"][0] for name, modes in grid.items()
        }
        get_strategy(strategy_name, **params)


# ─────────────────────────────────────────────────────────────────────
# A session strategy must say when its timeframe cannot work
# ─────────────────────────────────────────────────────────────────────

def _bars_at(freq, n=300):
    rng = np.random.default_rng(1)
    close = 1.1000 + np.cumsum(rng.normal(0, 0.0008, n))
    spread = np.abs(rng.normal(0.0004, 0.0002, n))
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq=freq, tz="UTC"),
        "open": close, "high": close + spread,
        "low": close - spread, "close": close,
        "volume": rng.integers(50, 500, n),
    })


def test_orb_warns_on_a_daily_timeframe(caplog):
    """The regression: it held forever and said nothing.

    ORB only signals inside a window opening minutes after a session
    open. Daily bars are timestamped 00:00, so no window ever matches and
    every candle returns HOLD — previously with no log at any level, so a
    D1 bot looked identical to one waiting for a setup.
    """
    from strategies.opening_range_strategy import OpeningRangeBreakoutStrategy

    strategy = OpeningRangeBreakoutStrategy()
    with caplog.at_level("WARNING"):
        strategy.analyze(_bars_at("1D"), "EURUSD")

    assert any("cannot" in r.message.lower() or "coarse" in r.message.lower()
               for r in caplog.records), caplog.text


def test_orb_warns_only_once(caplog):
    """Per candle would flood the log — the bot runs this every tick."""
    from strategies.opening_range_strategy import OpeningRangeBreakoutStrategy

    strategy = OpeningRangeBreakoutStrategy()
    df = _bars_at("1D")
    with caplog.at_level("WARNING"):
        for _ in range(5):
            strategy.analyze(df, "EURUSD")

    warnings_logged = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings_logged) == 1


def test_orb_is_quiet_on_an_intraday_timeframe(caplog):
    """The timeframe it is designed for must not warn."""
    from strategies.opening_range_strategy import OpeningRangeBreakoutStrategy

    strategy = OpeningRangeBreakoutStrategy()
    with caplog.at_level("WARNING"):
        strategy.analyze(_bars_at("15min"), "EURUSD")

    assert not [r for r in caplog.records if r.levelname == "WARNING"]


def test_orb_still_returns_hold_not_an_error_on_daily(caplog):
    """Warning, not raising — a coarse timeframe is a misconfiguration,
    not a crash, and the bot must keep running."""
    from strategies.opening_range_strategy import OpeningRangeBreakoutStrategy

    signal = OpeningRangeBreakoutStrategy().analyze(_bars_at("1D"), "EURUSD")
    assert signal.action.value == "HOLD"
