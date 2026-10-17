"""Tests for which open positions a bot may manage.

Ownership is consequential: the trading loop applies breakeven, the
time-stop and Quick Exit to every position it claims, and writes the
close to the trade-history DB.  Two bots claiming one position means it
gets force-closed and double-counted.
"""

from __future__ import annotations

import pytest

import multi_bot
from multi_bot import BotInstance, MultiBotManager

MAGIC = 123456


class _StubHistoryDB:
    """No-op stand-in so tests never touch data/bot_history.db."""

    def __getattr__(self, _name):
        return lambda *a, **kw: None


@pytest.fixture
def manager(monkeypatch, config):
    monkeypatch.setattr(multi_bot, "BotHistoryDB", _StubHistoryDB)
    config.magic_number = MAGIC
    config.adopt_untagged_positions = False
    return MultiBotManager(
        config=config, connector=None, market_data=None,
        session_mgr=None, order_mgr=None, risk_mgr=None,
    )


def _add(manager, symbol="EURUSD", strategy="ema"):
    return manager.add_bot(
        symbol=symbol, strategy_name=strategy, timeframe_key="M5",
    )


# ── Decisive comment match ───────────────────────────────────────────

def test_own_comment_is_owned(manager, mt5):
    bot = _add(manager)
    pos = mt5.Position(ticket=1, comment=f"Bot#{bot.bot_id}|ema", magic=MAGIC)
    assert manager._owns_position(bot, pos)


def test_other_bots_comment_is_not_owned(manager, mt5):
    bot = _add(manager)
    other = _add(manager)
    pos = mt5.Position(ticket=1, comment=f"Bot#{other.bot_id}|rsi", magic=MAGIC)
    assert not manager._owns_position(bot, pos)


# ── The regression this guards ───────────────────────────────────────

def test_two_bots_on_one_symbol_do_not_both_claim(manager, mt5):
    """The core bug: an untagged position was claimed by every bot.

    Both bots trade EURUSD and neither opened this position, so neither
    may manage it — previously both did, so both applied the time-stop
    and both recorded its close.
    """
    bot_a = _add(manager, "EURUSD")
    bot_b = _add(manager, "EURUSD")
    pos = mt5.Position(ticket=99, comment="", magic=MAGIC)

    claimants = [b for b in (bot_a, bot_b) if manager._owns_position(b, pos)]
    assert claimants == []


def test_tracked_ticket_survives_a_mangled_comment(manager, mt5):
    """A broker that rewrites the comment must not orphan our own trade."""
    bot_a = _add(manager, "EURUSD")
    bot_b = _add(manager, "EURUSD")
    bot_a.active_tickets.append(99)
    pos = mt5.Position(ticket=99, comment="xx-broker-rewrote-this", magic=MAGIC)

    assert manager._owns_position(bot_a, pos)
    assert not manager._owns_position(bot_b, pos)


# ── Adoption of pre-restart positions ────────────────────────────────

def test_sole_bot_adopts_its_magic_position(manager, mt5):
    """Keeps positions opened before a restart under management."""
    bot = _add(manager, "EURUSD")
    pos = mt5.Position(ticket=5, comment="", magic=MAGIC)
    assert manager._owns_position(bot, pos)


def test_foreign_magic_is_not_adopted(manager, mt5):
    """Another EA's trade on the same symbol is not ours to close."""
    bot = _add(manager, "EURUSD")
    pos = mt5.Position(ticket=5, comment="", magic=999999)
    assert not manager._owns_position(bot, pos)


def test_manual_position_not_adopted_by_default(manager, mt5):
    """A hand-placed trade must not inherit the bot's time-stop."""
    bot = _add(manager, "EURUSD")
    pos = mt5.Position(ticket=7, comment="", magic=0)
    assert not manager._owns_position(bot, pos)


def test_manual_position_adopted_when_opted_in(manager, mt5):
    bot = _add(manager, "EURUSD")
    manager._adopt_untagged = True
    pos = mt5.Position(ticket=7, comment="", magic=0)
    assert manager._owns_position(bot, pos)


def test_opt_in_adoption_still_requires_a_sole_bot(manager, mt5):
    """Even opted in, two bots must not both adopt one manual trade."""
    bot_a = _add(manager, "EURUSD")
    bot_b = _add(manager, "EURUSD")
    manager._adopt_untagged = True
    pos = mt5.Position(ticket=7, comment="", magic=0)

    claimants = [b for b in (bot_a, bot_b) if manager._owns_position(b, pos)]
    assert claimants == []


def test_bots_on_different_symbols_each_adopt_their_own(manager, mt5):
    eur = _add(manager, "EURUSD")
    gbp = _add(manager, "GBPUSD")
    pos = mt5.Position(ticket=11, symbol="EURUSD", comment="", magic=MAGIC)
    assert manager._owns_position(eur, pos)
    # GBP bot is sole-on-symbol for GBPUSD, but this position is only ever
    # offered to it by positions_get(symbol=...) for its own symbol.
    assert manager._sole_bot_on_symbol(gbp)
