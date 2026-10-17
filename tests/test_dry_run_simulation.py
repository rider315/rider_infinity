"""Tests for dry-run position simulation.

A dry-run bot places no real order, so ``active_tickets`` stayed empty
and the per-bot position limit never applied: the bot opened a fresh
"trade" on every candle and its trade count bore no relation to what it
would have done live.
"""

from __future__ import annotations

import time

import pandas as pd
import pytest

import multi_bot
from multi_bot import MultiBotManager


class _StubHistoryDB:
    def __getattr__(self, _name):
        return lambda *a, **kw: None


class _StubMarketData:
    def get_symbol_info(self, symbol):
        return {"point": 0.00001, "digits": 5}


@pytest.fixture
def manager(monkeypatch, config, tmp_path):
    monkeypatch.setattr(multi_bot, "BotHistoryDB", _StubHistoryDB)
    config.max_trade_hours = 48
    mgr = MultiBotManager(
        config=config, connector=None, market_data=_StubMarketData(),
        session_mgr=None, order_mgr=None, risk_mgr=None,
    )
    # Never write to the project's data/bot_configs.json —
    # add_bot() persists, and that file is the operator's live fleet.
    mgr._persist_path = tmp_path / "bot_configs.json"
    return mgr


@pytest.fixture
def bot(manager):
    b = manager.add_bot(
        symbol="EURUSD", strategy_name="ema", timeframe_key="M5",
        dry_run=True,
    )
    return b


def _bar_frame(high, low, close=None):
    close = close if close is not None else (high + low) / 2
    return pd.DataFrame([{
        "open": close, "high": high, "low": low,
        "close": close, "volume": 100,
    }])


def _sim(bot, direction="BUY", entry=1.1000, sl=1.0980, tp=1.1040, age_h=0.0):
    bot.sim_positions.append({
        "ticket": -1, "type": direction, "entry": entry,
        "sl": sl, "tp": tp, "lot": 0.01,
        "open_time": time.time() - age_h * 3600,
    })


def test_simulated_position_survives_an_untouched_candle(manager, bot, caplog):
    _sim(bot)
    manager._sync_dry_run_positions(bot, _bar_frame(1.1010, 1.0995), manager.logger)
    assert len(bot.sim_positions) == 1


def test_simulated_take_profit_closes_the_position(manager, bot):
    _sim(bot)
    manager._sync_dry_run_positions(bot, _bar_frame(1.1050, 1.1000), manager.logger)
    assert bot.sim_positions == []


def test_simulated_stop_loss_closes_the_position(manager, bot):
    _sim(bot)
    manager._sync_dry_run_positions(bot, _bar_frame(1.1000, 1.0970), manager.logger)
    assert bot.sim_positions == []


def test_simulated_sell_resolves_in_the_right_direction(manager, bot):
    _sim(bot, direction="SELL", entry=1.1000, sl=1.1020, tp=1.0960)
    # Price falls to the target.
    manager._sync_dry_run_positions(bot, _bar_frame(1.1005, 1.0950), manager.logger)
    assert bot.sim_positions == []


def test_simulated_position_hits_the_time_stop(manager, bot):
    _sim(bot, age_h=49)
    manager._sync_dry_run_positions(bot, _bar_frame(1.1010, 1.0995), manager.logger)
    assert bot.sim_positions == []


def test_time_stop_disabled_leaves_an_old_position_open(manager, bot):
    manager.config.max_trade_hours = 0
    _sim(bot, age_h=9999)
    manager._sync_dry_run_positions(bot, _bar_frame(1.1010, 1.0995), manager.logger)
    assert len(bot.sim_positions) == 1


def test_live_bot_is_untouched_by_the_simulator(manager):
    live = manager.add_bot(
        symbol="EURUSD", strategy_name="ema", timeframe_key="M5",
        dry_run=False,
    )
    _sim(live)
    manager._sync_dry_run_positions(live, _bar_frame(1.1050, 1.0950), manager.logger)
    # A live bot's real positions are reconciled from MT5, not from candles.
    assert len(live.sim_positions) == 1


def test_empty_frame_is_handled(manager, bot):
    _sim(bot)
    manager._sync_dry_run_positions(bot, pd.DataFrame(), manager.logger)
    assert len(bot.sim_positions) == 1


def test_dry_run_position_counts_against_the_per_bot_cap(manager, bot):
    """The regression: the cap had no effect at all in dry-run."""
    cap = manager.config.max_positions_per_bot
    assert cap >= 1

    _sim(bot)
    open_count = len(bot.active_tickets) + len(bot.sim_positions)
    assert open_count >= cap, (
        "a dry-run bot at its simulated limit must be throttled the same "
        "way the live bot would be"
    )
