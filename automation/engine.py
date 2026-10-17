"""
engine.py — Core Automation Engine for Rider Infinity.

Ties together triggers, conditions, and actions into automation rules
that are evaluated each trading iteration.
"""

from __future__ import annotations

import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from .triggers import (
    BaseTrigger,
    DrawdownThresholdTrigger,
    VolatilitySpikeTrigger,
    PnLThresholdTrigger,
)
from .conditions import BaseCondition
from .actions import BaseAction, PauseTradingAction, CloseAllPositionsAction, LogAction
from utils import utc_now

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# Data model
# ----------------------------------------------------------------------

@dataclass
class AutomationRule:
    """
    A single automation rule: trigger → conditions → actions.

    Parameters
    ----------
    name : str
        Human-readable rule name, e.g. ``'Emergency Stop'``.
    trigger : BaseTrigger
        Fires when market / account condition is met.
    conditions : list[BaseCondition]
        ALL must pass before actions execute.
    actions : list[BaseAction]
        Executed in order when trigger fires and conditions pass.
    enabled : bool
        Whether the rule is active.
    one_shot : bool
        If True, disable after first execution.
    """

    name: str
    trigger: BaseTrigger
    conditions: list[BaseCondition] = field(default_factory=list)
    actions: list[BaseAction] = field(default_factory=list)
    enabled: bool = True
    one_shot: bool = False

    # Auto-populated
    rule_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=utc_now)
    last_executed: datetime | None = None
    execution_count: int = 0


# ----------------------------------------------------------------------
# Automation engine
# ----------------------------------------------------------------------

class AutomationEngine:
    """
    Evaluates automation rules each iteration and executes matching
    actions.

    Usage
    -----
    >>> engine = AutomationEngine()
    >>> engine.add_rule(rule)
    >>> executed = engine.evaluate(context)
    """

    def __init__(self) -> None:
        self._rules: dict[str, AutomationRule] = {}
        self._execution_log: deque[dict[str, Any]] = deque(maxlen=500)
        logger.info("AutomationEngine initialised.")

    # ------------------------------------------------------------------
    # Rule management
    # ------------------------------------------------------------------

    def add_rule(self, rule: AutomationRule) -> None:
        """Register an automation rule."""
        self._rules[rule.rule_id] = rule
        logger.info("Rule added: '%s' (id=%s)", rule.name, rule.rule_id)

    def remove_rule(self, rule_id: str) -> bool:
        """Remove a rule by ID.  Returns True if found."""
        if rule_id in self._rules:
            name = self._rules.pop(rule_id).name
            logger.info("Rule removed: '%s' (id=%s)", name, rule_id)
            return True
        return False

    def enable_rule(self, rule_id: str) -> None:
        """Enable a rule."""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = True
            logger.info("Rule enabled: '%s'", self._rules[rule_id].name)

    def disable_rule(self, rule_id: str) -> None:
        """Disable a rule."""
        if rule_id in self._rules:
            self._rules[rule_id].enabled = False
            logger.info("Rule disabled: '%s'", self._rules[rule_id].name)

    def get_rules(self) -> list[AutomationRule]:
        """Return all rules."""
        return list(self._rules.values())

    def get_rule(self, rule_id: str) -> AutomationRule | None:
        """Return a specific rule or None."""
        return self._rules.get(rule_id)

    # ------------------------------------------------------------------
    # Evaluation loop
    # ------------------------------------------------------------------

    def evaluate(self, context: dict) -> list[str]:
        """
        Run all enabled rules against the current context.

        Parameters
        ----------
        context : dict
            Runtime state (prices, PnL, positions, etc.).

        Returns
        -------
        list[str]
            IDs of rules that were executed.
        """
        executed_ids: list[str] = []

        for rule in list(self._rules.values()):
            if not rule.enabled:
                continue

            # 1. Check trigger
            try:
                triggered = rule.trigger.check(context)
            except Exception:
                logger.exception(
                    "Trigger error in rule '%s' (%s).",
                    rule.name, rule.rule_id,
                )
                continue

            if not triggered:
                continue

            # 2. Check ALL conditions
            all_conditions_met = True
            for cond in rule.conditions:
                try:
                    if not cond.check(context):
                        all_conditions_met = False
                        break
                except Exception:
                    logger.exception(
                        "Condition error in rule '%s' (%s), condition=%s.",
                        rule.name, rule.rule_id, cond.name,
                    )
                    all_conditions_met = False
                    break

            if not all_conditions_met:
                logger.debug(
                    "Rule '%s' triggered but conditions not met.", rule.name,
                )
                continue

            # 3. Execute all actions
            logger.info(
                "AUTOMATION: executing rule '%s' (%s).", rule.name, rule.rule_id,
            )
            action_results = []
            for action in rule.actions:
                try:
                    result = action.execute(context)
                    action_results.append({
                        "action": action.name,
                        "success": result.success,
                        "message": result.message,
                    })
                    logger.info(
                        "  Action '%s': %s — %s",
                        action.name,
                        "OK" if result.success else "FAIL",
                        result.message,
                    )
                except Exception:
                    logger.exception(
                        "  Action error: '%s' in rule '%s'.",
                        action.name, rule.name,
                    )
                    action_results.append({
                        "action": action.name,
                        "success": False,
                        "message": "Exception raised.",
                    })

            # 4. Update rule state
            now = utc_now()
            rule.last_executed = now
            rule.execution_count += 1

            if rule.one_shot:
                rule.enabled = False
                logger.info(
                    "One-shot rule '%s' disabled after execution.", rule.name,
                )

            # 5. Log execution
            self._execution_log.append({
                "rule_id": rule.rule_id,
                "rule_name": rule.name,
                "timestamp": now.isoformat(),
                "action_results": action_results,
            })

            # Publish event if event bus is available
            try:
                from event_bus import emit_sync, EventChannel
                emit_sync(
                    EventChannel.AUTOMATION,
                    "rule_executed",
                    {
                        "rule_id": rule.rule_id,
                        "rule_name": rule.name,
                        "actions": [a["action"] for a in action_results],
                    },
                    source="AutomationEngine",
                )
            except ImportError:
                pass

            executed_ids.append(rule.rule_id)

        return executed_ids

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_execution_log(self) -> list[dict]:
        """Return recent execution log entries."""
        return list(self._execution_log)

    # ------------------------------------------------------------------
    # Default rules factory
    # ------------------------------------------------------------------

    @staticmethod
    def create_default_rules() -> list[AutomationRule]:
        """
        Return sensible default automation rules.

        Returns
        -------
        list[AutomationRule]
            Pre-configured rules for common scenarios.
        """
        rules = []

        # 1. Emergency Stop — 5 % drawdown → close all + pause 60 min
        rules.append(AutomationRule(
            name="Emergency Stop",
            trigger=DrawdownThresholdTrigger(max_drawdown_pct=5.0),
            conditions=[],
            actions=[
                CloseAllPositionsAction(),
                PauseTradingAction(duration_minutes=60),
                LogAction(message="EMERGENCY STOP triggered — 5% drawdown.", level="CRITICAL"),
            ],
            one_shot=True,
        ))

        # 2. High Volatility Pause — ATR spike 3× → pause 15 min
        rules.append(AutomationRule(
            name="High Volatility Pause",
            trigger=VolatilitySpikeTrigger(atr_multiplier=3.0),
            conditions=[],
            actions=[
                PauseTradingAction(duration_minutes=15),
                LogAction(message="Volatility spike detected — pausing 15 min.", level="WARNING"),
            ],
        ))

        # 3. Daily Target Reached — PnL > $500 → pause 24 h
        rules.append(AutomationRule(
            name="Daily Target Reached",
            trigger=PnLThresholdTrigger(threshold=500.0, direction="above"),
            conditions=[],
            actions=[
                PauseTradingAction(duration_minutes=1440),
                LogAction(message="Daily profit target reached! Pausing.", level="INFO"),
            ],
            one_shot=True,
        ))

        return rules
