"""
news_analyzer.py - AI-powered forex news analysis engine.

Uses Anthropic's Claude API to parse financial news articles and
extract structured trading intelligence: sentiment, impact scores,
affected currency pairs, and suggested actions.

Key features
------------
* Single-article and batch (up to 5) analysis modes.
* Exponential-backoff retry logic for transient API errors.
* Token-usage tracking with estimated cost reporting.
* Guaranteed non-throwing interface — always returns a valid result.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import anthropic

logger = logging.getLogger(__name__)

# ── Defaults & limits ────────────────────────────────────────────────────
_MAX_BATCH_SIZE: int = 5
_BASE_BACKOFF_SECONDS: float = 1.0


class ClaudeNewsAnalyzer:
    """Analyse financial news via Anthropic's Claude API.

    Parameters
    ----------
    api_key : str
        Anthropic API key.
    model : str
        Model identifier.  Empty (the default) auto-selects from
        :attr:`_PREFERRED_MODELS` based on what the key can reach.
    max_retries : int
        Maximum number of retry attempts on transient API errors.
    """

    # ── System prompt ────────────────────────────────────────────────────
    SYSTEM_PROMPT: str = (
        "You are a senior forex market analyst at a top-tier investment bank "
        "with 20+ years of experience.\n"
        "Your role is to analyze financial news articles and extract "
        "actionable trading intelligence.\n\n"
        "You must respond with ONLY valid JSON — no markdown, no explanation, "
        "no code fences.\n\n"
        "Analysis guidelines:\n"
        "- Focus on how news impacts FOREX currency pairs specifically\n"
        "- Consider both direct and indirect effects (e.g., strong US jobs "
        "data → USD strength → bearish EURUSD)\n"
        "- Rate central bank decisions and economic data releases as HIGH "
        "impact\n"
        "- Rate geopolitical events based on their potential to create "
        "risk-on/risk-off flows\n"
        "- Be conservative with confidence scores — only assign >0.8 for "
        "clear, unambiguous catalysts\n"
        "- Filter out clickbait, opinion pieces, and stale news (lower "
        "impact scores)\n"
        "- Consider currency correlations (e.g., AUD and NZD often move "
        "together)\n"
        "- Map news to specific forex pairs using standard notation "
        "(e.g., EURUSD, GBPUSD, USDJPY)\n\n"
        "IMPORTANT currency mapping rules:\n"
        "- If news is positive for USD → BEARISH for EURUSD, GBPUSD, "
        "AUDUSD, NZDUSD and BULLISH for USDJPY, USDCHF, USDCAD\n"
        "- If news is positive for EUR → BULLISH for EURUSD, EURGBP, "
        "EURJPY\n"
        "- If news is risk-off (fear/uncertainty) → BULLISH for USDJPY "
        "(yen safe haven), USDCHF (franc safe haven), XAUUSD (gold)\n"
        "- If news is risk-on (optimism) → BEARISH for safe havens, "
        "BULLISH for AUD, NZD pairs"
    )

    # ── Neutral fallback returned when analysis fails ────────────────────
    _NEUTRAL_FALLBACK: Dict[str, Any] = {
        "sentiment": "NEUTRAL",
        "confidence": 0.0,
        "impact_score": 0.0,
        "urgency": "LONG_TERM",
        "affected_symbols": [],
        "affected_sectors": [],
        "key_entities": [],
        "event_type": "OTHER",
        "reasoning": "Analysis unavailable — returned neutral fallback.",
        "suggested_action": "HOLD",
        "time_horizon": "Hours",
    }

    # ── Valid enum values for response validation ────────────────────────
    _VALID_SENTIMENTS = {"BULLISH", "BEARISH", "NEUTRAL"}
    _VALID_URGENCIES = {"IMMEDIATE", "SHORT_TERM", "LONG_TERM"}
    _VALID_EVENT_TYPES = {
        "CENTRAL_BANK", "ECONOMIC_DATA", "GEOPOLITICAL", "EARNINGS",
        "POLICY", "COMMODITY", "NATURAL_DISASTER", "OTHER",
    }
    _VALID_ACTIONS = {"BUY", "SELL", "HOLD", "WATCH"}
    _VALID_HORIZONS = {"Minutes", "Hours", "Days"}

    # ── Pricing, $ per million tokens (as of 2026-10) ────────────────────
    # Keyed by model-ID prefix, longest match first.  Previously a single
    # hard-coded $3/$15 pair — Sonnet 4.6's rate — which over-reported the
    # cost of every call on the current models by 50%.
    _PRICING_PER_MTOK: Dict[str, tuple] = {
        "claude-haiku-5-5": (0.10, 0.50),
        "claude-sonnet-5-5": (2.00, 10.00),
        "claude-sonnet-5": (2.00, 10.00),
        "claude-opus-5-5": (4.00, 20.00),
        "claude-opus-5": (5.00, 25.00),
        "claude-haiku-4-5": (1.00, 5.00),
        "claude-sonnet-4-6": (3.00, 15.00),
    }
    #: Used when the model is not in the table above, so an unknown or
    #: newer model reports *some* cost rather than zero.
    _FALLBACK_PRICING_PER_MTOK: tuple = (3.00, 15.00)

    # ------------------------------------------------------------------ #
    #  Construction                                                       #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        api_key: str,
        model: str = "",
        max_retries: int = 3,
    ) -> None:
        self.client: anthropic.Anthropic = anthropic.Anthropic(api_key=api_key)
        self.max_retries: int = max_retries

        # Auto-detect the best available model if none specified
        self.model: str = model or self._detect_model()

        # Usage counters
        self.total_calls: int = 0
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0

        logger.info(
            "ClaudeNewsAnalyzer initialised — model=%s, max_retries=%d",
            self.model, self.max_retries,
        )

    #: Models to use, most preferred first.  Sonnet is the default tier
    #: because this is bulk headline classification — high call volume,
    #: short structured output — where Sonnet's cost/quality is the right
    #: trade and Opus would be several times the price for no gain.
    #: Override by passing ``model=`` explicitly.
    _PREFERRED_MODELS: tuple = (
        "claude-sonnet-5-5",
        "claude-haiku-5-5",
        "claude-sonnet-5",
    )

    def _detect_model(self) -> str:
        """Pick the best model this API key can actually reach.

        Walks :attr:`_PREFERRED_MODELS` in order and returns the first one
        the account has access to.  If none of them are available — an
        older account, or a newer catalogue — it returns the first
        preference anyway and lets the API report the real error on the
        first call.

        .. note::
           The previous version fell back to "any model whose ID contains
           'sonnet', taking ``sorted(ids)[0]``".  Sorting IDs
           alphabetically is not sorting by recency: ``claude-3-5-sonnet``
           sorts before ``claude-sonnet-5-5``, so the fallback
           systematically picked the *oldest* available Sonnet. Its
           preference list was also stale (Sonnet 4.6 era), so on a
           current account the list missed and that fallback is what
           actually chose the model.
        """
        try:
            available_ids = {m.id for m in self.client.models.list().data}
        except Exception as exc:
            logger.warning(
                "Could not list models (%s) — using %s.",
                exc, self._PREFERRED_MODELS[0],
            )
            return self._PREFERRED_MODELS[0]

        logger.debug("Models available to this key: %s", sorted(available_ids))
        for candidate in self._PREFERRED_MODELS:
            if candidate in available_ids:
                logger.info("Auto-selected model: %s", candidate)
                return candidate

        logger.warning(
            "None of the preferred models %s are available to this key. "
            "Falling back to %s; pass model= explicitly to choose another.",
            list(self._PREFERRED_MODELS), self._PREFERRED_MODELS[0],
        )
        return self._PREFERRED_MODELS[0]

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def analyze(
        self,
        headline: str,
        summary: str = "",
        source: str = "",
    ) -> Dict[str, Any]:
        """Analyse a single news article and return structured trading intel.

        Parameters
        ----------
        headline : str
            Article headline (required).
        summary : str
            Optional body / summary text.
        source : str
            Optional source name (e.g. ``"Reuters"``).

        Returns
        -------
        dict
            Structured analysis with sentiment, confidence, impact_score,
            urgency, affected_symbols, affected_sectors, key_entities,
            event_type, reasoning, suggested_action, and time_horizon.
            Falls back to a neutral result on any unrecoverable error.
        """
        user_message = self._build_single_prompt(headline, summary, source)
        raw = self._call_api(user_message)

        if raw is None:
            logger.warning(
                "API call returned None for headline='%s'; using fallback.",
                headline,
            )
            return dict(self._NEUTRAL_FALLBACK)

        parsed = self._parse_single_response(raw)
        if parsed is None:
            logger.warning(
                "JSON parse failed for headline='%s'; using fallback.",
                headline,
            )
            return dict(self._NEUTRAL_FALLBACK)

        return self._sanitize(parsed)

    def analyze_batch(
        self,
        articles: List[Dict[str, str]],
    ) -> List[Dict[str, Any]]:
        """Analyse a batch of articles in a single Claude call.

        More token-efficient than calling :meth:`analyze` individually.
        The batch is capped at :data:`_MAX_BATCH_SIZE` articles; any
        overflow is silently truncated.

        Parameters
        ----------
        articles : list[dict]
            Each dict must have a ``headline`` key; ``summary`` and
            ``source`` are optional.

        Returns
        -------
        list[dict]
            One analysis dict per article, in the same order.  Falls
            back to individual analysis if the batch response cannot be
            parsed.
        """
        if not articles:
            return []

        batch = articles[:_MAX_BATCH_SIZE]

        user_message = self._build_batch_prompt(batch)
        raw = self._call_api(user_message)

        if raw is not None:
            parsed = self._parse_batch_response(raw, len(batch))
            if parsed is not None:
                return [self._sanitize(item) for item in parsed]

        # Fallback: analyse each article individually.
        logger.info(
            "Batch parse failed for %d articles; falling back to "
            "individual analysis.",
            len(batch),
        )
        return [
            self.analyze(
                headline=a.get("headline", ""),
                summary=a.get("summary", ""),
                source=a.get("source", ""),
            )
            for a in batch
        ]

    def get_usage_stats(self) -> Dict[str, Any]:
        """Return cumulative API usage statistics.

        Returns
        -------
        dict
            Keys: ``total_calls``, ``total_input_tokens``,
            ``total_output_tokens``, ``model``, ``estimated_cost_usd``.
        """
        in_rate, out_rate = self._rates_per_mtok()
        cost = (
            self.total_input_tokens * in_rate / 1_000_000
            + self.total_output_tokens * out_rate / 1_000_000
        )
        return {
            "total_calls": self.total_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "model": self.model,
            "estimated_cost_usd": round(cost, 6),
        }

    def _rates_per_mtok(self) -> tuple:
        """Return ``(input, output)`` $ per million tokens for this model.

        Matched on the longest ID prefix so a dated snapshot
        (``claude-sonnet-5-5-20260401``) prices like its base model.
        """
        for prefix in sorted(self._PRICING_PER_MTOK, key=len, reverse=True):
            if self.model.startswith(prefix):
                return self._PRICING_PER_MTOK[prefix]
        logger.debug(
            "No pricing entry for %s — estimating at $%.2f/$%.2f per MTok.",
            self.model, *self._FALLBACK_PRICING_PER_MTOK,
        )
        return self._FALLBACK_PRICING_PER_MTOK

    # ------------------------------------------------------------------ #
    #  Prompt builders                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_single_prompt(
        headline: str,
        summary: str = "",
        source: str = "",
    ) -> str:
        """Build the user-message for a single-article analysis."""
        parts = [
            "Analyze the following financial news article for forex "
            "trading implications.\n",
            f"Headline: {headline}",
        ]
        if summary:
            parts.append(f"Summary: {summary}")
        if source:
            parts.append(f"Source: {source}")

        parts.append(
            "\nRespond with a JSON object containing these exact fields:\n"
            "{\n"
            '  "sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",\n'
            '  "confidence": <float 0.0-1.0>,\n'
            '  "impact_score": <float 0.0-1.0>,\n'
            '  "urgency": "IMMEDIATE" | "SHORT_TERM" | "LONG_TERM",\n'
            '  "affected_symbols": [<list of forex pairs>],\n'
            '  "affected_sectors": [<list of sectors>],\n'
            '  "key_entities": [<list of key organizations/people>],\n'
            '  "event_type": "CENTRAL_BANK" | "ECONOMIC_DATA" | '
            '"GEOPOLITICAL" | "EARNINGS" | "POLICY" | "COMMODITY" | '
            '"NATURAL_DISASTER" | "OTHER",\n'
            '  "reasoning": "<string explanation>",\n'
            '  "suggested_action": "BUY" | "SELL" | "HOLD" | "WATCH",\n'
            '  "time_horizon": "Minutes" | "Hours" | "Days"\n'
            "}"
        )
        return "\n".join(parts)

    @staticmethod
    def _build_batch_prompt(articles: List[Dict[str, str]]) -> str:
        """Build the user-message for a batch analysis."""
        lines = [
            "Analyze the following financial news articles for forex "
            "trading implications.\n"
        ]
        for idx, article in enumerate(articles, start=1):
            lines.append(f"--- Article {idx} ---")
            lines.append(f"Headline: {article.get('headline', '')}")
            if article.get("summary"):
                lines.append(f"Summary: {article['summary']}")
            if article.get("source"):
                lines.append(f"Source: {article['source']}")
            lines.append("")

        lines.append(
            "Respond with a JSON **array** of objects (one per article, "
            "in the same order).  Each object must contain these exact "
            "fields:\n"
            "{\n"
            '  "sentiment": "BULLISH" | "BEARISH" | "NEUTRAL",\n'
            '  "confidence": <float 0.0-1.0>,\n'
            '  "impact_score": <float 0.0-1.0>,\n'
            '  "urgency": "IMMEDIATE" | "SHORT_TERM" | "LONG_TERM",\n'
            '  "affected_symbols": [<list of forex pairs>],\n'
            '  "affected_sectors": [<list of sectors>],\n'
            '  "key_entities": [<list of key organizations/people>],\n'
            '  "event_type": "CENTRAL_BANK" | "ECONOMIC_DATA" | '
            '"GEOPOLITICAL" | "EARNINGS" | "POLICY" | "COMMODITY" | '
            '"NATURAL_DISASTER" | "OTHER",\n'
            '  "reasoning": "<string explanation>",\n'
            '  "suggested_action": "BUY" | "SELL" | "HOLD" | "WATCH",\n'
            '  "time_horizon": "Minutes" | "Hours" | "Days"\n'
            "}"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    #  Claude API call with retries                                       #
    # ------------------------------------------------------------------ #

    def _call_api(self, user_message: str) -> Optional[str]:
        """Send *user_message* to Claude and return the raw text response.

        Implements exponential backoff for rate-limit errors and retries
        for transient API failures.  Returns ``None`` on total failure.
        """
        last_error: Optional[Exception] = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=[{
                        "type": "text",
                        "text": self.SYSTEM_PROMPT,
                        # Correct placement for a cache breakpoint: the
                        # system prompt is the only stable prefix (the
                        # article text is volatile and comes after it).
                        # At ~400 tokens it is currently *below* the
                        # 512-token minimum cacheable prefix, so this is
                        # a no-op today — it starts paying off by itself
                        # if the guidelines above grow past that.
                        "cache_control": {"type": "ephemeral"},
                    }],
                    # Classifying a headline into a fixed JSON schema does
                    # not need deep reasoning, and this runs on every
                    # article that comes in. The current models default to
                    # high effort, which would bill far more thinking than
                    # this task can use.
                    output_config={"effort": "low"},
                    messages=[{"role": "user", "content": user_message}],
                )

                # Track usage
                self.total_calls += 1
                if response.usage:
                    self.total_input_tokens += response.usage.input_tokens
                    self.total_output_tokens += response.usage.output_tokens

                # A safety classifier can decline a request: that arrives
                # as a normal HTTP 200 with stop_reason "refusal" and no
                # text block, so it has to be checked before the content
                # is read.
                if response.stop_reason == "refusal":
                    details = getattr(response, "stop_details", None)
                    logger.warning(
                        "Claude declined to analyse this article "
                        "(category=%s). Returning neutral.",
                        getattr(details, "category", None),
                    )
                    return None

                # Pick the text block rather than content[0]. Thinking is
                # on by default on the current models, so the first block
                # is often a thinking block, which has no .text at all —
                # reading content[0].text raised AttributeError.
                text = next(
                    (
                        block.text for block in response.content
                        if getattr(block, "type", None) == "text"
                    ),
                    None,
                )
                if text:
                    return text

                logger.warning(
                    "Claude returned no text block (attempt %d, "
                    "stop_reason=%s).", attempt, response.stop_reason,
                )

            except anthropic.RateLimitError as exc:
                last_error = exc
                backoff = _BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Rate-limited by Anthropic API (attempt %d/%d). "
                    "Backing off %.1fs. Error: %s",
                    attempt, self.max_retries, backoff, exc,
                )
                time.sleep(backoff)

            except anthropic.APIConnectionError as exc:
                last_error = exc
                backoff = _BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                logger.warning(
                    "Connection error to Anthropic API (attempt %d/%d). "
                    "Backing off %.1fs. Error: %s",
                    attempt, self.max_retries, backoff, exc,
                )
                time.sleep(backoff)

            except anthropic.APIError as exc:
                last_error = exc
                logger.error(
                    "Anthropic API error (attempt %d/%d): %s",
                    attempt, self.max_retries, exc,
                )
                # Non-retriable status codes: bail out immediately.
                if hasattr(exc, "status_code") and exc.status_code in (
                    400, 401, 403, 404,
                ):
                    break
                backoff = _BASE_BACKOFF_SECONDS * (2 ** (attempt - 1))
                time.sleep(backoff)

            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.exception(
                    "Unexpected error calling Claude (attempt %d/%d).",
                    attempt, self.max_retries,
                )
                break  # Unknown errors are not retried

        logger.error(
            "All %d API attempts exhausted. Last error: %s",
            self.max_retries, last_error,
        )
        return None

    # ------------------------------------------------------------------ #
    #  Response parsing                                                   #
    # ------------------------------------------------------------------ #

    def _parse_single_response(self, raw: str) -> Optional[Dict[str, Any]]:
        """Parse a single-article JSON response from Claude."""
        try:
            data = json.loads(self._strip_code_fences(raw))
            if isinstance(data, dict):
                return data
            # Claude occasionally wraps a single result in a list.
            if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
                return data[0]
            logger.warning("Unexpected JSON structure: %s", type(data).__name__)
        except json.JSONDecodeError as exc:
            logger.error("JSON decode error for single response: %s", exc)
        return None

    def _parse_batch_response(
        self, raw: str, expected_count: int,
    ) -> Optional[List[Dict[str, Any]]]:
        """Parse a batch JSON array response from Claude.

        Returns ``None`` if the parsed result is not a list of dicts
        whose length matches *expected_count*.
        """
        try:
            data = json.loads(self._strip_code_fences(raw))
            if isinstance(data, list) and len(data) == expected_count:
                if all(isinstance(item, dict) for item in data):
                    return data
            logger.warning(
                "Batch response length mismatch: expected %d, got %s.",
                expected_count,
                len(data) if isinstance(data, list) else type(data).__name__,
            )
        except json.JSONDecodeError as exc:
            logger.error("JSON decode error for batch response: %s", exc)
        return None

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        """Remove Markdown code fences if the model includes them."""
        stripped = text.strip()
        if stripped.startswith("```"):
            # Remove opening fence (possibly with language tag)
            first_newline = stripped.index("\n")
            stripped = stripped[first_newline + 1:]
        if stripped.endswith("```"):
            stripped = stripped[:-3]
        return stripped.strip()

    # ------------------------------------------------------------------ #
    #  Sanitisation                                                       #
    # ------------------------------------------------------------------ #

    def _sanitize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalise and clamp the parsed analysis to expected schema.

        Missing or invalid fields are filled with safe defaults so
        downstream code never encounters unexpected types.
        """
        result: Dict[str, Any] = {}

        # Enum-like strings
        sentiment = str(data.get("sentiment", "NEUTRAL")).upper()
        result["sentiment"] = (
            sentiment if sentiment in self._VALID_SENTIMENTS else "NEUTRAL"
        )

        urgency = str(data.get("urgency", "LONG_TERM")).upper()
        result["urgency"] = (
            urgency if urgency in self._VALID_URGENCIES else "LONG_TERM"
        )

        event_type = str(data.get("event_type", "OTHER")).upper()
        result["event_type"] = (
            event_type if event_type in self._VALID_EVENT_TYPES else "OTHER"
        )

        action = str(data.get("suggested_action", "HOLD")).upper()
        result["suggested_action"] = (
            action if action in self._VALID_ACTIONS else "HOLD"
        )

        # time_horizon is title-cased
        horizon = str(data.get("time_horizon", "Hours"))
        # Normalise to title-case for matching
        horizon_title = horizon.strip().title()
        result["time_horizon"] = (
            horizon_title if horizon_title in self._VALID_HORIZONS else "Hours"
        )

        # Floats clamped to [0, 1]
        result["confidence"] = self._clamp_float(data.get("confidence", 0.0))
        result["impact_score"] = self._clamp_float(data.get("impact_score", 0.0))

        # Lists
        result["affected_symbols"] = self._ensure_str_list(
            data.get("affected_symbols"),
        )
        result["affected_sectors"] = self._ensure_str_list(
            data.get("affected_sectors"),
        )
        result["key_entities"] = self._ensure_str_list(
            data.get("key_entities"),
        )

        # Free-text
        result["reasoning"] = str(data.get("reasoning", ""))

        return result

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _clamp_float(value: Any, lo: float = 0.0, hi: float = 1.0) -> float:
        """Convert *value* to a float clamped to [lo, hi]."""
        try:
            return max(lo, min(hi, float(value)))
        except (TypeError, ValueError):
            return lo

    @staticmethod
    def _ensure_str_list(value: Any) -> List[str]:
        """Coerce *value* to a list of strings."""
        if isinstance(value, list):
            return [str(v) for v in value]
        return []
