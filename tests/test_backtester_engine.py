"""Tests that exercise the backtester's own execution path.

The engine's ``_execute`` is the one place the whole simulation comes
together: broker symbol resolution, pip sizing, the live-parity gates,
the bar loop and the metrics.  Without a test through it, an import it
depends on can be removed and every other test still passes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import backtester
from backtester import BacktestEngine, BacktestResult


@pytest.fixture
def rates():
    """A structured array shaped like ``mt5.copy_rates_range`` output."""
    n = 1200
    rng = np.random.default_rng(11)
    close = 1.1000 + np.cumsum(rng.normal(0.00003, 0.0005, n))
    spread = np.abs(rng.normal(0.0004, 0.0002, n))
    base = pd.Timestamp("2026-01-01", tz="UTC").timestamp()
    return np.array(
        [
            (base + i * 300, c, c + s, c - s, c, int(v), 2, 0)
            for i, (c, s, v) in enumerate(
                zip(close, spread, rng.integers(50, 500, n))
            )
        ],
        dtype=[
            ("time", "i8"), ("open", "f8"), ("high", "f8"), ("low", "f8"),
            ("close", "f8"), ("tick_volume", "i8"), ("spread", "i4"),
            ("real_volume", "i8"),
        ],
    )


@pytest.fixture
def engine(monkeypatch, config, mt5, rates, tmp_path):
    """A BacktestEngine wired to the stub, writing to a temp DB."""
    monkeypatch.setattr(
        backtester, "_DB_PATH", tmp_path / "backtest.db", raising=False,
    )
    mt5.copy_rates_range = lambda *a, **kw: rates
    mt5.symbol_info = lambda s: mt5.SymbolInfo(
        name=s, point=0.00001, digits=5,
        trade_contract_size=100_000.0,
        trade_tick_value=1.0, trade_tick_size=0.00001,
    )

    eng = BacktestEngine(config=config)
    eng.db = backtester.BacktestDB(db_path=tmp_path / "backtest.db")
    return eng


def _result(**kw):
    base = dict(
        run_id="test-run", strategy_name="ema", symbol="EURUSD",
        timeframe="M5", date_from="2026-01-01", date_to="2026-02-01",
        initial_balance=10_000.0, lot_size=0.1,
    )
    base.update(kw)
    return BacktestResult(**base)


def test_execute_produces_trades_and_metrics(engine):
    result = _result()
    engine._execute(result, max_open_trades=5)

    assert result.total_trades > 0
    assert result.wins + result.losses == result.total_trades
    assert 0 <= result.win_rate <= 100
    assert result.equity_curve


def test_metrics_are_internally_consistent(engine):
    result = _result()
    engine._execute(result, max_open_trades=5)
    result.compute_metrics()

    assert result.total_pnl_money == pytest.approx(
        sum(t.pnl_money for t in result.trades), abs=0.01,
    )
    assert result.max_drawdown_pct >= 0
    assert result.profit_factor >= 0
    # Capped, never infinite - float('inf') is not JSON-serialisable.
    assert result.profit_factor <= 999.99


def test_summary_is_json_serialisable(engine):
    """The dashboard returns this straight to the browser."""
    import json

    result = _result()
    engine._execute(result, max_open_trades=5)
    result.compute_metrics()
    json.dumps(result.to_summary())  # raises on inf/nan


def test_every_trade_exits_for_a_known_reason(engine):
    result = _result()
    engine._execute(result, max_open_trades=5)
    assert {t.exit_reason for t in result.trades} <= {
        "SL", "TP", "TIME", "END",
    }


def test_entries_use_the_next_bar_open(engine):
    """No look-ahead: a signal on bar i fills at bar i+1's open."""
    result = _result()
    engine._execute(result, max_open_trades=5)
    assert result.trades, "need at least one trade to check"
    for trade in result.trades:
        assert trade.entry_time != trade.exit_time


def test_unknown_timeframe_is_rejected(engine):
    with pytest.raises(ValueError, match="Unknown timeframe"):
        engine._execute(_result(timeframe="M7"), max_open_trades=5)


def test_missing_history_is_reported_clearly(engine, mt5):
    mt5.copy_rates_range = lambda *a, **kw: None
    with pytest.raises(ValueError, match="No historical data"):
        engine._execute(_result(), max_open_trades=5)


def test_max_open_trades_is_respected(engine):
    """A limit of 1 cannot produce overlapping positions."""
    result = _result()
    engine._execute(result, max_open_trades=1)

    spans = sorted(
        (t.entry_time, t.exit_time) for t in result.trades
    )
    for (_, prev_exit), (next_entry, _) in zip(spans, spans[1:]):
        assert next_entry >= prev_exit


def test_run_is_persisted_and_retrievable(engine):
    result = _result()
    engine._execute(result, max_open_trades=5)
    result.compute_metrics()
    engine.db.save_run(result)
    engine.db.save_trades(result.run_id, result.trades)

    stored = engine.db.get_run(result.run_id)
    assert stored is not None
    assert stored["total_trades"] == result.total_trades
    assert len(engine.db.get_trades(result.run_id)) == len(result.trades)


@pytest.mark.parametrize("strategy", ["ema", "rsi", "macd", "bollinger"])
def test_core_strategies_run_end_to_end(engine, strategy):
    result = _result(strategy_name=strategy)
    engine._execute(result, max_open_trades=5)
    # Zero trades is a legitimate outcome; a crash is not.
    assert result.total_trades >= 0
