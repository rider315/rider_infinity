"""Tests for indicator agreement and the market-open calendar.

Two subtle disagreements lived here: the live engine sized stops from a
simple-moving-average ATR while the strategies gated entries on Wilder's,
and ``risk_manager`` carried a second hand-rolled forex calendar that
contradicted ``SessionManager``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from session_manager import SessionManager
from strategies import indicators as ind
from utils import compute_atr


def _ohlc(n=100, seed=7):
    """A deterministic OHLC frame with genuine variation in range."""
    rng = np.random.default_rng(seed)
    close = 1.1000 + np.cumsum(rng.normal(0, 0.0004, n))
    spread = np.abs(rng.normal(0.0006, 0.0004, n))
    return pd.DataFrame({
        "open": close,
        "high": close + spread,
        "low": close - spread,
        "close": close,
        "volume": rng.integers(50, 500, n),
    })


# ─────────────────────────────────────────────────────────────────────
# ATR agreement
# ─────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("period", [7, 14, 21])
def test_live_atr_matches_the_strategy_atr(period):
    """The regression.

    ``utils.compute_atr`` sizes live stops; ``indicators.atr`` gates
    strategy entries. They must be the same number on the same candle,
    or a stop is sized from volatility the strategy never saw.
    """
    df = _ohlc()
    assert compute_atr(df, period) == pytest.approx(
        float(ind.atr(df, period).iloc[-1]), rel=1e-9,
    )


def test_atr_uses_wilder_not_a_simple_average():
    """Wilder's smoothing must differ from an SMA on real data.

    If this passes trivially the comparison above proves nothing.
    """
    df = _ohlc()
    tr = ind.true_range(df)
    sma = float(tr.rolling(14).mean().iloc[-1])
    assert compute_atr(df, 14) != pytest.approx(sma, rel=1e-6)


def test_atr_is_positive_for_moving_prices():
    assert compute_atr(_ohlc(), 14) > 0


def test_atr_of_flat_prices_is_zero():
    flat = pd.DataFrame({
        "open": [1.1] * 50, "high": [1.1] * 50,
        "low": [1.1] * 50, "close": [1.1] * 50,
    })
    assert compute_atr(flat, 14) == 0.0


def test_atr_returns_zero_rather_than_nan_on_short_data():
    """A NaN here would propagate into a live stop price."""
    short = _ohlc(n=5)
    assert compute_atr(short, 14) == 0.0


def test_atr_returns_zero_on_missing_columns():
    assert compute_atr(pd.DataFrame({"close": [1.0] * 50}), 14) == 0.0


def test_atr_of_none_is_zero():
    assert compute_atr(None, 14) == 0.0


# ─────────────────────────────────────────────────────────────────────
# Market calendar
# ─────────────────────────────────────────────────────────────────────

def _at(year, month, day, hour):
    return datetime(year, month, day, hour, tzinfo=timezone.utc)


@pytest.fixture
def sessions():
    return SessionManager()


# 2026-10-17 is a Saturday, 10-18 Sunday, 10-19 Monday.
@pytest.mark.parametrize("when", [
    _at(2026, 10, 17, 3),    # Saturday morning
    _at(2026, 10, 17, 23),   # Saturday night
    _at(2026, 10, 18, 12),   # Sunday midday
])
def test_forex_is_closed_at_the_weekend(sessions, when):
    assert not sessions.is_market_open(now=when, symbol="EURUSD")


def test_forex_is_open_midweek(sessions):
    assert sessions.is_market_open(now=_at(2026, 10, 21, 13), symbol="EURUSD")


@pytest.mark.parametrize("when", [
    _at(2026, 10, 17, 3), _at(2026, 10, 18, 12), _at(2026, 10, 21, 13),
])
def test_crypto_never_closes(sessions, when):
    assert sessions.is_market_open(now=when, symbol="BTCUSD")


def test_risk_manager_uses_the_same_calendar(config, mt5, monkeypatch):
    """The regression: risk_manager had its own, contradictory calendar.

    Its version treated Saturday before 22:00 UTC as open, treated all of
    Sunday as closed, and carried an unreachable ``hour < 0`` branch.
    """
    import risk_manager

    class _Connector:
        def get_account_info(self):
            return {"equity": 10_000.0, "balance": 10_000.0}

    sessions = SessionManager()
    rm = risk_manager.RiskManager(config, _Connector(), session_mgr=sessions)

    saturday = _at(2026, 10, 17, 3)
    monkeypatch.setattr(
        risk_manager, "datetime",
        type("_D", (), {"now": staticmethod(lambda tz=None: saturday)}),
    )

    can, reason = rm.can_trade("EURUSD", 0.01)
    assert not can
    assert "closed" in reason.lower()
    # And it agrees with the calendar it now delegates to.
    assert not sessions.is_market_open(now=saturday, symbol="EURUSD")
