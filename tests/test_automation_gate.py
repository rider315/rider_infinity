"""Tests for the global trading gate and the automation rules that use it.

``PauseTradingAction`` used to write ``trading_paused`` into the context
dict that ``AutomationEngine.evaluate`` is handed.  That dict is rebuilt
every iteration and nothing ever read the key, so the Emergency Stop rule
closed every open position and then let the bots re-enter immediately.
"""

from __future__ import annotations

import time

import pytest

import trade_manager
from automation.actions import PauseTradingAction
from automation.engine import AutomationEngine
from trade_manager import TradingGate


@pytest.fixture(autouse=True)
def _reset_global_gate():
    """The gate is process-global; never leak a pause between tests."""
    trade_manager.GATE.resume()
    yield
    trade_manager.GATE.resume()


# ── TradingGate ──────────────────────────────────────────────────────

def test_a_fresh_gate_allows_trading():
    gate = TradingGate()
    assert gate.block_reason() is None
    assert gate.is_paused is False


def test_pause_blocks_then_expires():
    gate = TradingGate()
    gate.pause(minutes=10, reason="drawdown")
    assert gate.is_paused

    reason = gate.block_reason()
    assert "drawdown" in reason
    assert "9m" in reason

    # Look at the gate from beyond the expiry.
    assert gate.block_reason(now=time.time() + 601) is None


def test_resume_clears_a_pause():
    gate = TradingGate()
    gate.pause(minutes=60, reason="emergency")
    gate.resume()
    assert gate.block_reason() is None


def test_a_longer_pause_extends_a_shorter_one():
    gate = TradingGate()
    gate.pause(minutes=5, reason="volatility")
    gate.pause(minutes=60, reason="emergency")
    assert "emergency" in gate.block_reason()


def test_a_shorter_pause_cannot_cut_short_a_longer_one():
    """A 15-minute volatility pause must not end a 60-minute stop early."""
    gate = TradingGate()
    gate.pause(minutes=60, reason="emergency")
    gate.pause(minutes=15, reason="volatility")

    assert "emergency" in gate.block_reason()
    # Still blocked well past the shorter pause's expiry.
    assert gate.block_reason(now=time.time() + 16 * 60) is not None


def test_zero_minutes_does_not_block():
    gate = TradingGate()
    gate.pause(minutes=0, reason="noop")
    assert gate.block_reason() is None


def test_to_dict_reports_a_clear_state():
    gate = TradingGate()
    assert gate.to_dict() == {"paused": False, "reason": "", "until": None}

    gate.pause(minutes=30, reason="emergency")
    snapshot = gate.to_dict()
    assert snapshot["paused"] is True
    assert snapshot["reason"] == "emergency"
    assert snapshot["until"].endswith("+00:00")


# ── PauseTradingAction ───────────────────────────────────────────────

def test_pause_action_arms_the_global_gate():
    """The regression: the action must have an effect outside the context."""
    assert trade_manager.GATE.block_reason() is None

    result = PauseTradingAction(duration_minutes=45).execute({})

    assert result.success
    assert trade_manager.GATE.block_reason() is not None


def test_pause_action_still_mirrors_into_context():
    context: dict = {}
    PauseTradingAction(duration_minutes=10).execute(context)
    assert context["trading_paused"] is True
    assert context["trading_paused_until"]


def test_pause_action_carries_a_reason_through():
    PauseTradingAction(duration_minutes=10).execute(
        {"pause_reason": "volatility spike"},
    )
    assert "volatility spike" in trade_manager.GATE.block_reason()


# ── End to end through the engine ────────────────────────────────────

def test_emergency_stop_rule_actually_pauses_the_fleet():
    """The default Emergency Stop rule, evaluated for real.

    Previously this closed the positions and left trading wide open.
    """
    engine = AutomationEngine()
    rules = {r.name: r for r in AutomationEngine.create_default_rules()}
    emergency = rules["Emergency Stop"]
    engine.add_rule(emergency)

    closed: list[str] = []

    class _OrderMgr:
        def close_all_positions(self, symbol=None):
            closed.append(symbol or "ALL")
            return 3

        def get_open_positions(self, symbol=None):
            return []

    executed = engine.evaluate({
        "drawdown_pct": 9.0,          # past the rule's 5% threshold
        "order_manager": _OrderMgr(),
        "symbol": "EURUSD",
    })

    assert executed == [emergency.rule_id]
    assert closed, "Emergency Stop should have closed open positions"
    assert trade_manager.GATE.is_paused, (
        "Emergency Stop must also block new entries, or the bots "
        "immediately re-enter what was just closed"
    )


def test_engine_records_execution_counts():
    """The dashboard reports these; a throwaway engine always showed 0."""
    engine = AutomationEngine()
    for rule in AutomationEngine.create_default_rules():
        engine.add_rule(rule)

    before = sum(r.execution_count for r in engine.get_rules())
    engine.evaluate({"drawdown_pct": 99.0, "symbol": "EURUSD"})
    after = sum(r.execution_count for r in engine.get_rules())

    assert after > before
    assert engine.get_execution_log()


def test_one_shot_rule_disables_itself():
    engine = AutomationEngine()
    rules = {r.name: r for r in AutomationEngine.create_default_rules()}
    emergency = rules["Emergency Stop"]
    assert emergency.one_shot
    engine.add_rule(emergency)

    engine.evaluate({"drawdown_pct": 99.0, "symbol": "EURUSD"})
    assert emergency.enabled is False

    # A second pass must not fire it again.
    assert engine.evaluate({"drawdown_pct": 99.0, "symbol": "EURUSD"}) == []


def test_disabled_rule_never_fires():
    engine = AutomationEngine()
    rules = {r.name: r for r in AutomationEngine.create_default_rules()}
    emergency = rules["Emergency Stop"]
    engine.add_rule(emergency)
    engine.disable_rule(emergency.rule_id)

    assert engine.evaluate({"drawdown_pct": 99.0}) == []
    assert not trade_manager.GATE.is_paused
