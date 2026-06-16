"""
Evaluation Matrix — Multi‑Strategy Aggregation Engine
=======================================================

Inspired by OctoBot's evaluation matrix, this module collects
``EvalScore`` objects from multiple evaluators (strategies, indicators,
social signals, etc.) and aggregates them into a single ``TradeDecision``.

Three aggregation modes are supported:

* **WEIGHTED_AVERAGE** — ``Σ(score × weight × confidence) / Σ(weight × confidence)``
* **MAJORITY_VOTE** — direction with more weighted votes wins
* **UNANIMOUS** — all evaluators must agree on direction, else neutral

Usage
-----
>>> from evaluation_matrix import EvaluationMatrix, EvalScore
>>> matrix = EvaluationMatrix()
>>> matrix.submit_score(EvalScore(value=-0.6, confidence=0.8,
...                               evaluator_name="RSI", reason="oversold"))
>>> matrix.submit_score(EvalScore(value=-0.4, confidence=0.7,
...                               evaluator_name="MACD", reason="bullish cross"))
>>> decision = matrix.aggregate()
>>> decision.state
<EvaluatorState.LONG: 'LONG'>
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EvaluatorState(Enum):
    """Maps a continuous score to a discrete trading stance.

    Score boundaries
    ----------------
    * ``VERY_LONG``  — score ≤ −0.7
    * ``LONG``       — score ≤ −0.3
    * ``NEUTRAL``    — −0.3 < score < 0.3
    * ``SHORT``      — score ≥  0.3
    * ``VERY_SHORT`` — score ≥  0.7
    """

    VERY_LONG = "VERY_LONG"
    LONG = "LONG"
    NEUTRAL = "NEUTRAL"
    SHORT = "SHORT"
    VERY_SHORT = "VERY_SHORT"


class AggregationMode(Enum):
    """Strategy for combining multiple evaluator scores.

    Members
    -------
    WEIGHTED_AVERAGE
        Sum of ``score × weight × confidence`` divided by
        sum of ``weight × confidence``.
    MAJORITY_VOTE
        The direction (bullish / bearish) with more weighted
        votes determines the sign; magnitude is the margin.
    UNANIMOUS
        Returns a non‑zero score only if **all** evaluators
        agree on direction; otherwise returns 0.0 (neutral).
    """

    WEIGHTED_AVERAGE = "WEIGHTED_AVERAGE"
    MAJORITY_VOTE = "MAJORITY_VOTE"
    UNANIMOUS = "UNANIMOUS"


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------

@dataclass
class EvalScore:
    """A single evaluator's assessment of market conditions.

    Attributes
    ----------
    value : float
        Score in the range ``[-1.0, +1.0]``.
        * −1.0 → strong **buy** (bullish)
        *  0.0 → neutral
        * +1.0 → strong **sell** (bearish)
    confidence : float
        How confident the evaluator is, ``[0.0, 1.0]``.
    weight : float
        Pertinence / importance weight for this evaluator.
    evaluator_name : str
        Human‑readable name of the evaluator that produced this score.
    reason : str
        Free‑text explanation of why this score was given.
    sl_pips : float
        Suggested stop‑loss distance in pips (0 if not specified).
    tp_pips : float
        Suggested take‑profit distance in pips (0 if not specified).
    timestamp : datetime
        When this score was produced (auto‑set to *now* by default).
    """

    value: float
    confidence: float
    weight: float = 1.0
    evaluator_name: str = ""
    reason: str = ""
    sl_pips: float = 0.0
    tp_pips: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        self.value = max(-1.0, min(1.0, float(self.value)))
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.weight = max(0.0, float(self.weight))


@dataclass
class TradeDecision:
    """Aggregated output of the evaluation matrix.

    Attributes
    ----------
    state : EvaluatorState
        Discrete trading stance derived from ``final_score``.
    final_score : float
        Aggregated score in ``[-1.0, +1.0]``.
    confidence : float
        Aggregated confidence in ``[0.0, 1.0]``.
    contributing_scores : list[EvalScore]
        All individual scores that were combined.
    sl_pips : float
        Weighted‑average stop‑loss from contributors.
    tp_pips : float
        Weighted‑average take‑profit from contributors.
    reason : str
        Combined human‑readable explanation.
    timestamp : datetime
        When the decision was produced (auto‑set).
    """

    state: EvaluatorState
    final_score: float
    confidence: float
    contributing_scores: List[EvalScore] = field(default_factory=list)
    sl_pips: float = 0.0
    tp_pips: float = 0.0
    reason: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------

class EvaluationMatrix:
    """Collects evaluator scores and aggregates them into a ``TradeDecision``.

    Parameters
    ----------
    mode : AggregationMode
        How individual scores are combined (default: weighted average).
    min_confidence : float
        Scores with confidence below this threshold are silently
        discarded during aggregation (default: 0.3).

    Examples
    --------
    >>> matrix = EvaluationMatrix()
    >>> matrix.submit_score(EvalScore(value=-0.5, confidence=0.9,
    ...                               evaluator_name="RSI", reason="oversold"))
    >>> decision = matrix.aggregate()
    """

    def __init__(
        self,
        mode: AggregationMode = AggregationMode.WEIGHTED_AVERAGE,
        min_confidence: float = 0.3,
    ) -> None:
        self._mode = mode
        self._min_confidence = min_confidence
        self._scores: List[EvalScore] = []

    # -- Public API ---------------------------------------------------------

    def submit_score(self, score: EvalScore) -> None:
        """Add an evaluator score to the current evaluation round.

        Parameters
        ----------
        score : EvalScore
            The score to register.  Scores with zero weight are ignored.
        """
        if score.weight <= 0:
            logger.debug(
                "Ignoring score from %s (weight=%.2f)",
                score.evaluator_name,
                score.weight,
            )
            return
        self._scores.append(score)
        logger.debug(
            "Score submitted: %s → value=%.3f confidence=%.2f weight=%.2f",
            score.evaluator_name,
            score.value,
            score.confidence,
            score.weight,
        )

    def clear(self) -> None:
        """Reset all scores for the next evaluation round."""
        self._scores.clear()
        logger.debug("Evaluation matrix cleared")

    def get_scores(self) -> List[EvalScore]:
        """Return a copy of the currently submitted scores.

        Returns
        -------
        list[EvalScore]
        """
        return list(self._scores)

    def aggregate(self) -> TradeDecision:
        """Compute a final ``TradeDecision`` from all submitted scores.

        Scores whose confidence is below ``min_confidence`` are filtered
        out before aggregation.

        Returns
        -------
        TradeDecision
        """
        # Filter by minimum confidence
        eligible = [
            s for s in self._scores if s.confidence >= self._min_confidence
        ]

        if not eligible:
            logger.info(
                "No eligible scores (total=%d, min_confidence=%.2f) → NEUTRAL",
                len(self._scores),
                self._min_confidence,
            )
            return TradeDecision(
                state=EvaluatorState.NEUTRAL,
                final_score=0.0,
                confidence=0.0,
                contributing_scores=list(self._scores),
                reason="No eligible evaluator scores",
            )

        # Aggregate value
        dispatch = {
            AggregationMode.WEIGHTED_AVERAGE: self._weighted_average,
            AggregationMode.MAJORITY_VOTE: self._majority_vote,
            AggregationMode.UNANIMOUS: self._unanimous,
        }
        raw_score = dispatch[self._mode](eligible)
        final_score = max(-1.0, min(1.0, raw_score))

        # Aggregate confidence (weighted average of confidences)
        total_weight = sum(s.weight for s in eligible)
        agg_confidence = (
            sum(s.confidence * s.weight for s in eligible) / total_weight
            if total_weight > 0
            else 0.0
        )

        # Weighted average SL / TP (only from scores that specify them)
        sl_pips = self._weighted_avg_field(eligible, "sl_pips")
        tp_pips = self._weighted_avg_field(eligible, "tp_pips")

        # Combined reason
        reason_parts = [
            f"{s.evaluator_name}: {s.reason}" for s in eligible if s.reason
        ]
        combined_reason = " | ".join(reason_parts) if reason_parts else "—"

        state = self._compute_state(final_score)

        decision = TradeDecision(
            state=state,
            final_score=round(final_score, 4),
            confidence=round(agg_confidence, 4),
            contributing_scores=list(self._scores),
            sl_pips=round(sl_pips, 2),
            tp_pips=round(tp_pips, 2),
            reason=combined_reason,
        )

        logger.info(
            "Aggregated %d scores (mode=%s) → %s  score=%.4f  conf=%.2f",
            len(eligible),
            self._mode.value,
            state.value,
            final_score,
            agg_confidence,
        )
        return decision

    # -- Internal aggregation methods ---------------------------------------

    def _weighted_average(self, scores: List[EvalScore]) -> float:
        """Weighted average: Σ(value × weight × confidence) / Σ(weight × confidence).

        Parameters
        ----------
        scores : list[EvalScore]
            Eligible scores (already filtered by min_confidence).

        Returns
        -------
        float
        """
        numerator = sum(s.value * s.weight * s.confidence for s in scores)
        denominator = sum(s.weight * s.confidence for s in scores)
        if denominator == 0:
            return 0.0
        return numerator / denominator

    def _majority_vote(self, scores: List[EvalScore]) -> float:
        """Majority vote: direction with more weighted votes wins.

        Each score casts a weighted vote for its direction (sign).
        The result is the difference of bullish vs bearish vote mass,
        normalised to ``[-1.0, +1.0]``.

        Parameters
        ----------
        scores : list[EvalScore]
            Eligible scores.

        Returns
        -------
        float
        """
        bullish_weight = 0.0  # negative values → bullish
        bearish_weight = 0.0  # positive values → bearish

        for s in scores:
            vote_strength = s.weight * s.confidence
            if s.value < 0:
                bullish_weight += vote_strength
            elif s.value > 0:
                bearish_weight += vote_strength
            # value == 0 → abstain

        total = bullish_weight + bearish_weight
        if total == 0:
            return 0.0

        # Positive result → bearish, negative → bullish (matches value convention)
        return (bearish_weight - bullish_weight) / total

    def _unanimous(self, scores: List[EvalScore]) -> float:
        """Unanimous: returns 0 if any scores disagree on direction.

        All non‑neutral scores must share the same sign for a
        non‑zero result.  If they do, the weighted‑average is returned;
        otherwise 0.0 (neutral).

        Parameters
        ----------
        scores : list[EvalScore]
            Eligible scores.

        Returns
        -------
        float
        """
        non_neutral = [s for s in scores if s.value != 0.0]
        if not non_neutral:
            return 0.0

        signs = {1 if s.value > 0 else -1 for s in non_neutral}
        if len(signs) > 1:
            logger.debug("Unanimous check failed — mixed directions")
            return 0.0

        # All agree — fall back to weighted average for magnitude
        return self._weighted_average(scores)

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _compute_state(score: float) -> EvaluatorState:
        """Map a continuous score to a discrete ``EvaluatorState``.

        Parameters
        ----------
        score : float
            Aggregated score in ``[-1.0, +1.0]``.

        Returns
        -------
        EvaluatorState
        """
        if score <= -0.7:
            return EvaluatorState.VERY_LONG
        if score <= -0.3:
            return EvaluatorState.LONG
        if score >= 0.7:
            return EvaluatorState.VERY_SHORT
        if score >= 0.3:
            return EvaluatorState.SHORT
        return EvaluatorState.NEUTRAL

    @staticmethod
    def _weighted_avg_field(scores: List[EvalScore], attr: str) -> float:
        """Compute the weighted average of a numeric field across scores.

        Only scores where the field is non‑zero contribute.

        Parameters
        ----------
        scores : list[EvalScore]
            Eligible scores.
        attr : str
            Attribute name on ``EvalScore`` (e.g. ``"sl_pips"``).

        Returns
        -------
        float
        """
        contributors = [s for s in scores if getattr(s, attr, 0) != 0]
        if not contributors:
            return 0.0
        total_weight = sum(s.weight * s.confidence for s in contributors)
        if total_weight == 0:
            return 0.0
        return sum(
            getattr(s, attr) * s.weight * s.confidence for s in contributors
        ) / total_weight
