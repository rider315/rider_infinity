"""
automation — No-Code Automation Engine for Rider Infinity.

Provides a Trigger → Condition → Action pipeline for automated
responses to market events, account conditions, and time-based rules.
"""

from .triggers import (
    BaseTrigger,
    PriceThresholdTrigger,
    PnLThresholdTrigger,
    DrawdownThresholdTrigger,
    TimeBasedTrigger,
    PeriodicTrigger,
    VolatilitySpikeTrigger,
)
from .conditions import (
    BaseCondition,
    SessionActiveCondition,
    MinPositionsCondition,
    MaxPositionsCondition,
    SpreadBelowCondition,
)
from .actions import (
    BaseAction,
    ActionResult,
    PauseTradingAction,
    CloseAllPositionsAction,
    ReducePositionSizesAction,
    LogAction,
)
from .engine import AutomationEngine, AutomationRule
