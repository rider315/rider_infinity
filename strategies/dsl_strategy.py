"""
DSL Custom Strategy
===================

A Domain-Specific Language strategy that lets users define trading
rules as plain-text without writing Python.

Rule Syntax
-----------
.. code-block:: text

    # Comments start with #
    BUY WHEN RSI(14) < 30 AND EMA(12) > EMA(26)
    SELL WHEN RSI(14) > 70 AND MACD_HIST(12,26,9) < 0

Supported Indicators
--------------------
RSI(period), EMA(period), SMA(period), ATR(period),
MACD(f,s,sig), MACD_HIST(f,s,sig), MACD_SIGNAL(f,s,sig),
BB_UPPER(period,std), BB_LOWER(period,std), BB_MIDDLE(period,std),
CLOSE, OPEN, HIGH, LOW, VOLUME

Operators: ``<``, ``>``, ``<=``, ``>=``, ``==``
Logic: ``AND`` (conditions on the same line are AND-ed)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from .base import BaseStrategy, Signal, SignalAction, TriggerProximity

logger = logging.getLogger(__name__)


# ----------------------------------------------------------------------
# DSL data model
# ----------------------------------------------------------------------

@dataclass
class DSLCondition:
    """A single comparison: ``left_operand op right_operand``."""
    left_operand: str
    operator: str
    right_operand: str


@dataclass
class DSLRule:
    """One parsed rule line."""
    action: str          # 'BUY' or 'SELL'
    conditions: list[DSLCondition] = field(default_factory=list)
    raw_text: str = ""
    line_number: int = 0


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------

VALID_OPERATORS = {"<", ">", "<=", ">=", "=="}

KNOWN_INDICATORS = {
    "RSI", "EMA", "SMA", "ATR",
    "MACD", "MACD_HIST", "MACD_SIGNAL",
    "BB_UPPER", "BB_LOWER", "BB_MIDDLE",
}

PRICE_FIELDS = {"CLOSE", "OPEN", "HIGH", "LOW", "VOLUME"}

# Matches patterns like  RSI(14)  or  MACD_HIST(12,26,9)
_FUNC_RE = re.compile(
    r"([A-Z_]+)\(([^)]+)\)"
)


class DSLParser:
    """
    Parse and validate DSL rule text.

    Examples
    --------
    >>> parser = DSLParser()
    >>> rules = parser.parse("BUY WHEN RSI(14) < 30")
    >>> ok, errors = parser.validate(rules)
    """

    def parse(self, rules_text: str) -> list[DSLRule]:
        """
        Parse multi-line rule text into ``DSLRule`` objects.

        Parameters
        ----------
        rules_text : str
            The raw rule text.

        Returns
        -------
        list[DSLRule]
        """
        rules: list[DSLRule] = []

        for lineno, raw_line in enumerate(rules_text.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            upper = line.upper()
            if not (upper.startswith("BUY") or upper.startswith("SELL")):
                logger.warning("Line %d: skipped (no BUY/SELL): %s", lineno, line)
                continue

            parts = upper.split(None, 1)
            action = parts[0]

            # Remove the 'WHEN' keyword
            rest = parts[1] if len(parts) > 1 else ""
            if rest.startswith("WHEN "):
                rest = rest[5:]

            # Split by AND
            condition_strs = [c.strip() for c in rest.split(" AND ") if c.strip()]

            conditions: list[DSLCondition] = []
            for cond_str in condition_strs:
                parsed = self._parse_condition(cond_str)
                if parsed:
                    conditions.append(parsed)
                else:
                    logger.warning(
                        "Line %d: could not parse condition '%s'",
                        lineno, cond_str,
                    )

            rules.append(DSLRule(
                action=action,
                conditions=conditions,
                raw_text=raw_line,
                line_number=lineno,
            ))

        return rules

    def validate(self, rules: list[DSLRule]) -> tuple[bool, list[str]]:
        """
        Validate parsed rules for common errors.

        Returns
        -------
        tuple[bool, list[str]]
            ``(True, [])`` if valid, ``(False, [errors])`` otherwise.
        """
        errors: list[str] = []

        if not rules:
            errors.append("No rules defined.")
            return (False, errors)

        for rule in rules:
            if rule.action not in ("BUY", "SELL"):
                errors.append(
                    f"Line {rule.line_number}: invalid action '{rule.action}'."
                )
            if not rule.conditions:
                errors.append(
                    f"Line {rule.line_number}: rule has no conditions."
                )
            for cond in rule.conditions:
                if cond.operator not in VALID_OPERATORS:
                    errors.append(
                        f"Line {rule.line_number}: invalid operator "
                        f"'{cond.operator}'."
                    )

        return (len(errors) == 0, errors)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_condition(text: str) -> DSLCondition | None:
        """Parse a single condition like ``RSI(14) < 30``."""
        for op in ("<=", ">=", "==", "<", ">"):
            if f" {op} " in text:
                left, right = text.split(f" {op} ", 1)
                return DSLCondition(
                    left_operand=left.strip(),
                    operator=op,
                    right_operand=right.strip(),
                )
        return None


# ----------------------------------------------------------------------
# Strategy
# ----------------------------------------------------------------------

class DSLStrategy(BaseStrategy):
    """
    User-defined trading rules via a simple DSL.

    Parameters
    ----------
    rules_text : str
        The DSL rules (multi-line string).
    sl_pips : float
        Stop-loss in pips.
    tp_pips : float
        Take-profit in pips.
    """

    def __init__(
        self,
        rules_text: str = "",
        sl_pips: float = 50.0,
        tp_pips: float = 100.0,
    ) -> None:
        super().__init__(sl_pips=sl_pips, tp_pips=tp_pips)
        self.rules_text = rules_text
        self._parser = DSLParser()
        self._rules = self._parser.parse(rules_text)

        ok, errors = self._parser.validate(self._rules)
        if not ok:
            for err in errors:
                logger.error("DSL validation: %s", err)

    @property
    def name(self) -> str:
        return "DSL Custom Strategy"

    # ------------------------------------------------------------------
    # Analyze
    # ------------------------------------------------------------------

    def analyze(self, df: pd.DataFrame, symbol: str) -> Signal:
        """Evaluate DSL rules against current market data."""
        if not self._rules or len(df) < 30:
            return self._hold(symbol)

        # Cache indicator values for this call
        cache: dict[str, Any] = {}

        for rule in self._rules:
            all_met = True
            for cond in rule.conditions:
                try:
                    left_val = self._evaluate_operand(cond.left_operand, df, cache)
                    right_val = self._evaluate_operand(cond.right_operand, df, cache)

                    if left_val is None or right_val is None:
                        all_met = False
                        break

                    if not self._compare(left_val, cond.operator, right_val):
                        all_met = False
                        break
                except Exception:
                    logger.debug(
                        "DSL condition error: %s %s %s",
                        cond.left_operand, cond.operator, cond.right_operand,
                    )
                    all_met = False
                    break

            if all_met:
                action = (
                    SignalAction.BUY if rule.action == "BUY"
                    else SignalAction.SELL
                )
                reason = f"DSL rule matched: {rule.raw_text.strip()}"
                logger.info("[%s] %s — %s", self.name, symbol, reason)
                return Signal(
                    action=action,
                    symbol=symbol,
                    sl_pips=self.sl_pips,
                    tp_pips=self.tp_pips,
                    confidence=0.7,
                    reason=reason,
                )

        return self._hold(symbol)

    def get_proximity(self, df: pd.DataFrame, symbol: str) -> TriggerProximity:
        """Show distance to nearest DSL rule trigger."""
        if not self._rules or len(df) < 30:
            return TriggerProximity()

        cache: dict[str, Any] = {}
        best_proximity = 0.0
        best_rule = None

        for rule in self._rules:
            met_count = 0
            for cond in rule.conditions:
                try:
                    left_val = self._evaluate_operand(cond.left_operand, df, cache)
                    right_val = self._evaluate_operand(cond.right_operand, df, cache)
                    if left_val is not None and right_val is not None:
                        if self._compare(left_val, cond.operator, right_val):
                            met_count += 1
                except Exception:
                    pass

            if rule.conditions:
                proximity = met_count / len(rule.conditions)
                if proximity > best_proximity:
                    best_proximity = proximity
                    best_rule = rule

        if best_rule is None:
            return TriggerProximity()

        return TriggerProximity(
            proximity_pct=round(best_proximity, 4),
            direction=best_rule.action,
            label=f"DSL: {best_proximity*100:.0f}% conditions met",
            detail=best_rule.raw_text.strip(),
        )

    def evaluate(self, df: pd.DataFrame, symbol: str) -> "EvalScore":
        """Return an evaluation score for the evaluation matrix."""
        from evaluation_matrix import EvalScore

        signal = self.analyze(df, symbol)

        if signal.action == SignalAction.BUY:
            value = -signal.confidence
        elif signal.action == SignalAction.SELL:
            value = signal.confidence
        else:
            value = 0.0

        return EvalScore(
            value=value,
            confidence=signal.confidence,
            weight=1.0,
            evaluator_name=self.name,
            reason=signal.reason,
            sl_pips=signal.sl_pips,
            tp_pips=signal.tp_pips,
        )

    # ------------------------------------------------------------------
    # Indicator evaluation
    # ------------------------------------------------------------------

    def _evaluate_operand(
        self, operand: str, df: pd.DataFrame, cache: dict
    ) -> float | None:
        """Evaluate an operand — either a number or an indicator."""
        operand = operand.strip()

        # Numeric literal?
        try:
            return float(operand)
        except ValueError:
            pass

        # Cached?
        if operand in cache:
            return cache[operand]

        # Price field?
        upper = operand.upper()
        if upper in PRICE_FIELDS:
            col = upper.lower()
            if col in df.columns:
                val = float(df[col].iloc[-1])
                cache[operand] = val
                return val
            return None

        # Function call?
        match = _FUNC_RE.match(upper)
        if match:
            func_name = match.group(1)
            args = [a.strip() for a in match.group(2).split(",")]
            val = self._calculate_indicator(func_name, args, df)
            if val is not None:
                cache[operand] = val
            return val

        return None

    def _calculate_indicator(
        self, name: str, args: list[str], df: pd.DataFrame,
    ) -> float | None:
        """Calculate a technical indicator from the DataFrame."""
        close = df["close"]

        try:
            if name == "RSI":
                period = int(args[0])
                delta = close.diff()
                gain = delta.where(delta > 0, 0.0).rolling(period).mean()
                loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
                rs = gain / loss.replace(0, float("nan"))
                rsi = 100.0 - (100.0 / (1.0 + rs))
                return float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else None

            if name == "EMA":
                period = int(args[0])
                ema = close.ewm(span=period, adjust=False).mean()
                return float(ema.iloc[-1])

            if name == "SMA":
                period = int(args[0])
                sma = close.rolling(period).mean()
                val = sma.iloc[-1]
                return float(val) if not pd.isna(val) else None

            if name == "ATR":
                period = int(args[0])
                high, low = df["high"], df["low"]
                prev_close = close.shift(1)
                tr = pd.concat([
                    high - low,
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ], axis=1).max(axis=1)
                atr = tr.rolling(period).mean()
                val = atr.iloc[-1]
                return float(val) if not pd.isna(val) else None

            if name in ("MACD", "MACD_HIST", "MACD_SIGNAL"):
                fast, slow, sig = int(args[0]), int(args[1]), int(args[2])
                ema_fast = close.ewm(span=fast, adjust=False).mean()
                ema_slow = close.ewm(span=slow, adjust=False).mean()
                macd_line = ema_fast - ema_slow
                signal_line = macd_line.ewm(span=sig, adjust=False).mean()
                histogram = macd_line - signal_line

                if name == "MACD":
                    return float(macd_line.iloc[-1])
                elif name == "MACD_SIGNAL":
                    return float(signal_line.iloc[-1])
                else:  # MACD_HIST
                    return float(histogram.iloc[-1])

            if name in ("BB_UPPER", "BB_LOWER", "BB_MIDDLE"):
                period = int(args[0])
                std_mult = float(args[1]) if len(args) > 1 else 2.0
                sma = close.rolling(period).mean()
                std = close.rolling(period).std()

                if name == "BB_UPPER":
                    val = (sma + std_mult * std).iloc[-1]
                elif name == "BB_LOWER":
                    val = (sma - std_mult * std).iloc[-1]
                else:
                    val = sma.iloc[-1]
                return float(val) if not pd.isna(val) else None

        except (IndexError, ValueError, KeyError):
            logger.debug("Failed to calculate %s(%s).", name, args)
            return None

        return None

    @staticmethod
    def _compare(left: float, op: str, right: float) -> bool:
        """Apply a comparison operator."""
        if op == "<":
            return left < right
        if op == ">":
            return left > right
        if op == "<=":
            return left <= right
        if op == ">=":
            return left >= right
        if op == "==":
            return abs(left - right) < 1e-9
        return False
