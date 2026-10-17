"""Tests for the robustness validation entry points.

Genetic search, walk-forward analysis and Monte Carlo were fully
implemented but unreachable: nothing in the code-base called them, and
the only reference was a CLI menu that printed the import statements the
user would need to wire them up themselves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import validation
from backtest_core import Trade, compute_metrics, run_single


def _bars(n=1600, seed=3):
    """Deterministic OHLCV bars with a mild trend plus noise."""
    rng = np.random.default_rng(seed)
    close = 1.1000 + np.cumsum(rng.normal(0.00002, 0.0005, n))
    spread = np.abs(rng.normal(0.0004, 0.0002, n))
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
        "open": close,
        "high": close + spread,
        "low": close - spread,
        "close": close,
        "volume": rng.integers(50, 500, n),
    })


@pytest.fixture
def market():
    return validation.MarketSlice(
        _bars(), symbol="EURUSD", point=0.00001, digits=5, pip_value=10.0,
    )


# ─────────────────────────────────────────────────────────────────────
# The kernel the tools run on
# ─────────────────────────────────────────────────────────────────────

def test_run_single_produces_metrics(market, config):
    metrics = run_single(
        strategy_name="ema", params={"fast_period": 9, "slow_period": 21},
        df=market.df, symbol=market.symbol, point=market.point,
        digits=market.digits, pip_value_per_lot=market.pip_value,
        lot_size=0.1, initial_balance=10_000.0,
        default_sl=config.default_sl_pips, default_tp=config.default_tp_pips,
    )
    assert metrics["total_trades"] > 0
    assert set(metrics) >= {
        "total_trades", "win_rate", "profit_factor",
        "max_drawdown_pct", "sharpe_ratio",
    }


def test_compute_metrics_on_no_trades_is_all_zero():
    metrics = compute_metrics([], 10_000.0, 10_000.0)
    assert metrics["total_trades"] == 0
    assert metrics["profit_factor"] == 0


def test_compute_metrics_counts_wins_and_losses():
    trades = [
        Trade(10, 100.0, "TP", "t0", "t1"),
        Trade(-5, -50.0, "SL", "t1", "t2"),
        Trade(20, 200.0, "TP", "t2", "t3"),
    ]
    metrics = compute_metrics(trades, 10_000.0, 10_250.0)
    assert metrics["total_trades"] == 3
    assert metrics["wins"] == 2
    assert metrics["losses"] == 1
    assert metrics["win_rate"] == pytest.approx(66.7, abs=0.1)
    assert metrics["profit_factor"] == pytest.approx(6.0)


# ─────────────────────────────────────────────────────────────────────
# backtest_fn builders
# ─────────────────────────────────────────────────────────────────────

def test_backtest_fn_takes_params_only(market, config):
    """The shape GeneticOptimizer calls."""
    fn = validation.make_backtest_fn("ema", market, config)
    metrics = fn({"fast_period": 5, "slow_period": 20})
    assert "profit_factor" in metrics


def test_windowed_backtest_fn_takes_params_and_a_window(market, config):
    """The shape WalkForwardAnalyzer calls."""
    fn = validation.make_windowed_backtest_fn("ema", market, config)
    metrics = fn({"fast_period": 5, "slow_period": 20}, market.df.iloc[:600])
    assert "profit_factor" in metrics


def test_different_params_give_different_results(market, config):
    """If they didn't, the search would be meaningless."""
    fn = validation.make_backtest_fn("ema", market, config)
    a = fn({"fast_period": 5, "slow_period": 20})
    b = fn({"fast_period": 20, "slow_period": 100})
    assert a != b


# ─────────────────────────────────────────────────────────────────────
# Genetic search
# ─────────────────────────────────────────────────────────────────────

def test_genetic_search_runs_and_ranks(monkeypatch, market, config):
    monkeypatch.setattr(validation, "fetch_slice",
                        lambda *a, **kw: market)
    out = validation.run_genetic(
        "ema", "EURUSD", "M5", "2026-01-01", "2026-02-01",
        config=config, population_size=6, generations=2,
    )
    assert out["method"] == "genetic"
    assert out["evaluations"] == 12
    assert out["search_space"] > out["evaluations"]  # that's the point
    assert out["results"]
    metrics = [r["profit_factor"] for r in out["results"]]
    assert metrics == sorted(metrics, reverse=True)


def test_genetic_reports_progress(monkeypatch, market, config):
    monkeypatch.setattr(validation, "fetch_slice", lambda *a, **kw: market)
    seen: list[int] = []
    validation.run_genetic(
        "ema", "EURUSD", "M5", "2026-01-01", "2026-02-01",
        config=config, population_size=4, generations=3,
        callback=lambda pct, best, gen: seen.append(gen),
    )
    assert seen == [1, 2, 3]


def test_genetic_rejects_an_unknown_strategy(config):
    with pytest.raises(ValueError, match="No parameter grid"):
        validation.run_genetic(
            "not-a-strategy", "EURUSD", "M5", "2026-01-01", "2026-02-01",
            config=config,
        )


# ─────────────────────────────────────────────────────────────────────
# Walk-forward
# ─────────────────────────────────────────────────────────────────────

def test_walk_forward_scores_out_of_sample(monkeypatch, market, config):
    monkeypatch.setattr(validation, "fetch_slice", lambda *a, **kw: market)
    result = validation.run_walk_forward(
        "ema", "EURUSD", "M5", "2026-01-01", "2026-03-01",
        config=config, in_sample_bars=500, out_of_sample_bars=200,
        step_bars=400, max_combos=4,
    )
    assert result.windows
    for window in result.windows:
        assert "in_sample_metric" in window
        assert "out_of_sample_metric" in window
        # Each out-of-sample window must start after its in-sample one.
        assert window["window_end"] > window["window_start"]


def test_walk_forward_robustness_ratio_is_oos_over_is(monkeypatch, market,
                                                      config):
    monkeypatch.setattr(validation, "fetch_slice", lambda *a, **kw: market)
    result = validation.run_walk_forward(
        "ema", "EURUSD", "M5", "2026-01-01", "2026-03-01",
        config=config, in_sample_bars=500, out_of_sample_bars=200,
        step_bars=400, max_combos=4,
    )
    if result.avg_in_sample_metric > 0:
        assert result.robustness_ratio == pytest.approx(
            result.avg_out_of_sample_metric / result.avg_in_sample_metric,
            abs=0.001,
        )
    assert result.is_robust == (result.robustness_ratio > 0.5)


def test_walk_forward_refuses_too_short_a_range(monkeypatch, config):
    short = validation.MarketSlice(
        _bars(n=300), "EURUSD", 0.00001, 5, 10.0,
    )
    monkeypatch.setattr(validation, "fetch_slice", lambda *a, **kw: short)
    with pytest.raises(ValueError, match="at least 700 bars"):
        validation.run_walk_forward(
            "ema", "EURUSD", "M5", "2026-01-01", "2026-01-05",
            config=config, in_sample_bars=500, out_of_sample_bars=200,
        )


def test_walk_forward_caps_the_grid(monkeypatch, market, config):
    """Cost is combos x windows, so an uncapped deep grid never finishes."""
    monkeypatch.setattr(validation, "fetch_slice", lambda *a, **kw: market)
    calls: list[dict] = []
    real = validation.make_windowed_backtest_fn

    def _counting(strategy, mkt, cfg, **kw):
        fn = real(strategy, mkt, cfg, **kw)

        def _wrapped(params, window):
            calls.append(params)
            return fn(params, window)

        return _wrapped

    monkeypatch.setattr(validation, "make_windowed_backtest_fn", _counting)
    validation.run_walk_forward(
        "ema", "EURUSD", "M5", "2026-01-01", "2026-03-01",
        config=config, in_sample_bars=500, out_of_sample_bars=200,
        step_bars=600, max_combos=3,
    )
    distinct = {tuple(sorted(p.items())) for p in calls}
    assert len(distinct) <= 3


# ─────────────────────────────────────────────────────────────────────
# Monte Carlo
# ─────────────────────────────────────────────────────────────────────

def _trade_rows(pnls):
    return [{"pnl_money": p} for p in pnls]


def test_monte_carlo_reports_percentiles():
    result = validation.run_monte_carlo(
        _trade_rows([50, -30, 80, -20, 60, -40, 70, -10]),
        initial_balance=10_000.0, num_simulations=200,
    )
    assert result.num_simulations == 200
    assert result.percentile_5_return <= result.median_return
    assert result.median_return <= result.percentile_95_return
    assert result.worst_case_return <= result.percentile_5_return
    assert 0.0 <= result.probability_of_profit <= 100.0


def test_monte_carlo_sees_an_all_winning_series_as_certain():
    result = validation.run_monte_carlo(
        _trade_rows([10, 20, 30, 40]), num_simulations=100,
    )
    assert result.probability_of_profit == pytest.approx(100.0)


def test_monte_carlo_sees_an_all_losing_series_as_hopeless():
    result = validation.run_monte_carlo(
        _trade_rows([-10, -20, -30, -40]), num_simulations=100,
    )
    assert result.probability_of_profit == pytest.approx(0.0)


def test_monte_carlo_drawdown_depends_on_order():
    """The whole point: the same trades reordered risk a deeper drawdown."""
    result = validation.run_monte_carlo(
        _trade_rows([100, -90, 100, -90, 100, -90]),
        initial_balance=1000.0, num_simulations=400,
    )
    assert result.worst_case_max_drawdown >= result.median_max_drawdown


def test_monte_carlo_needs_at_least_two_trades():
    with pytest.raises(ValueError, match="at least 2"):
        validation.run_monte_carlo(_trade_rows([10]))


def test_monte_carlo_ignores_rows_with_no_pnl():
    rows = _trade_rows([10, 20]) + [{"pnl_money": None}, {}]
    result = validation.run_monte_carlo(rows, num_simulations=50)
    assert result.num_simulations == 50
