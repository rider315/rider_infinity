"""Tests for the Claude news analyser.

Covers model selection, response parsing and cost accounting.  The
`anthropic` package is stubbed, so these run without a key and spend
nothing.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest


# ── Stub the anthropic SDK before the analyser imports it ───────────

class _StubAPIError(Exception):
    def __init__(self, message="", status_code=None):
        super().__init__(message)
        self.status_code = status_code


class _StubRateLimitError(_StubAPIError):
    pass


class _StubConnectionError(_StubAPIError):
    pass


class _Block:
    def __init__(self, type_, text=None, thinking=None):
        self.type = type_
        if text is not None:
            self.text = text
        if thinking is not None:
            self.thinking = thinking


class _Usage:
    def __init__(self, input_tokens=100, output_tokens=50):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _Response:
    def __init__(self, content, stop_reason="end_turn", stop_details=None,
                 usage=None):
        self.content = content
        self.stop_reason = stop_reason
        self.stop_details = stop_details
        self.usage = usage or _Usage()


class _Models:
    def __init__(self, ids):
        self._ids = ids

    def list(self):
        return types.SimpleNamespace(
            data=[types.SimpleNamespace(id=i) for i in self._ids],
        )


class _Messages:
    def __init__(self):
        self.calls: list[dict] = []
        self.response: Any = None
        self.raises: Exception | None = None

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises is not None:
            raise self.raises
        return self.response


class _StubClient:
    def __init__(self, api_key=None, model_ids=()):
        self.models = _Models(list(model_ids))
        self.messages = _Messages()


@pytest.fixture(autouse=True)
def _stub_anthropic(monkeypatch):
    """Install a stub `anthropic` module and reload the analyser against it."""
    stub = types.ModuleType("anthropic")
    stub.Anthropic = lambda api_key=None: _StubClient(api_key)
    stub.APIError = _StubAPIError
    stub.APIStatusError = _StubAPIError
    stub.RateLimitError = _StubRateLimitError
    stub.APIConnectionError = _StubConnectionError
    monkeypatch.setitem(sys.modules, "anthropic", stub)
    for mod in list(sys.modules):
        if mod.startswith("news_intelligence"):
            monkeypatch.delitem(sys.modules, mod, raising=False)
    yield stub


@pytest.fixture
def analyzer_cls(_stub_anthropic):
    from news_intelligence.news_analyzer import ClaudeNewsAnalyzer
    return ClaudeNewsAnalyzer


def _make(analyzer_cls, model_ids, model=""):
    client = _StubClient(model_ids=model_ids)
    analyzer = analyzer_cls.__new__(analyzer_cls)
    analyzer.client = client
    analyzer.max_retries = 2
    analyzer.total_calls = 0
    analyzer.total_input_tokens = 0
    analyzer.total_output_tokens = 0
    analyzer.model = model or analyzer._detect_model()
    return analyzer


# ─────────────────────────────────────────────────────────────────────
# Model selection
# ─────────────────────────────────────────────────────────────────────

CURRENT = [
    "claude-3-5-sonnet-20240620",
    "claude-sonnet-4-6",
    "claude-sonnet-5-5",
    "claude-haiku-5-5",
    "claude-opus-5-5",
]


def test_picks_the_first_available_preference(analyzer_cls):
    a = _make(analyzer_cls, CURRENT)
    assert a.model == "claude-sonnet-5-5"


def test_does_not_pick_the_alphabetically_first_sonnet(analyzer_cls):
    """The regression.

    The old fallback took ``sorted(ids)[0]`` among IDs containing
    "sonnet". Alphabetical order is not recency, so it systematically
    chose the oldest Sonnet the account could see.
    """
    a = _make(analyzer_cls, CURRENT)
    assert a.model != "claude-3-5-sonnet-20240620"
    assert a.model != min(i for i in CURRENT if "sonnet" in i)


def test_falls_back_down_the_preference_list(analyzer_cls):
    """With no Sonnet 5.5, the next preference wins - not an old model."""
    a = _make(analyzer_cls, ["claude-3-5-sonnet-20240620", "claude-haiku-5-5"])
    assert a.model == "claude-haiku-5-5"


def test_unreachable_catalogue_uses_the_top_preference(analyzer_cls):
    a = _make(analyzer_cls, [])
    assert a.model == analyzer_cls._PREFERRED_MODELS[0]


def test_model_listing_failure_is_survivable(analyzer_cls, monkeypatch):
    client = _StubClient(model_ids=[])

    def _boom():
        raise RuntimeError("network down")

    client.models.list = _boom
    a = analyzer_cls.__new__(analyzer_cls)
    a.client = client
    assert a._detect_model() == analyzer_cls._PREFERRED_MODELS[0]


def test_preferred_models_are_current_generation(analyzer_cls):
    """Guards against the list going stale again."""
    for model_id in analyzer_cls._PREFERRED_MODELS:
        assert "-5" in model_id, f"{model_id} looks like an older generation"
        assert not model_id[-8:].isdigit(), (
            f"{model_id} carries a date suffix; use the bare model ID"
        )


# ─────────────────────────────────────────────────────────────────────
# Response handling
# ─────────────────────────────────────────────────────────────────────

def test_reads_the_text_block(analyzer_cls):
    a = _make(analyzer_cls, CURRENT)
    a.client.messages.response = _Response([_Block("text", text='{"ok": 1}')])
    assert a._call_api("hello") == '{"ok": 1}'


def test_skips_a_leading_thinking_block(analyzer_cls):
    """The regression: content[0].text crashed when thinking came first.

    Thinking is on by default on the current models, and a thinking
    block exposes .thinking, not .text - so reading content[0].text
    raised AttributeError on every call.
    """
    a = _make(analyzer_cls, CURRENT)
    a.client.messages.response = _Response([
        _Block("thinking", thinking="considering the headline"),
        _Block("text", text='{"sentiment": "BULLISH"}'),
    ])
    assert a._call_api("hello") == '{"sentiment": "BULLISH"}'


def test_refusal_returns_none_without_raising(analyzer_cls):
    """A declined request is HTTP 200 with no text block."""
    a = _make(analyzer_cls, CURRENT)
    a.client.messages.response = _Response(
        [], stop_reason="refusal",
        stop_details=types.SimpleNamespace(
            type="refusal", category="general_harms", explanation="",
        ),
    )
    assert a._call_api("hello") is None


def test_refusal_does_not_burn_retries(analyzer_cls):
    a = _make(analyzer_cls, CURRENT)
    a.client.messages.response = _Response([], stop_reason="refusal")
    a._call_api("hello")
    assert len(a.client.messages.calls) == 1


def test_empty_content_is_retried(analyzer_cls):
    a = _make(analyzer_cls, CURRENT)
    a.client.messages.response = _Response([])
    assert a._call_api("hello") is None
    assert len(a.client.messages.calls) == a.max_retries


def test_analyze_returns_the_neutral_fallback_on_refusal(analyzer_cls):
    a = _make(analyzer_cls, CURRENT)
    a.client.messages.response = _Response([], stop_reason="refusal")
    result = a.analyze(headline="Something the classifier declines")
    assert result["sentiment"] == "NEUTRAL"
    assert result["confidence"] == 0.0


# ─────────────────────────────────────────────────────────────────────
# Request shape
# ─────────────────────────────────────────────────────────────────────

def test_request_pins_low_effort(analyzer_cls):
    """Bulk headline classification must not bill high-effort thinking."""
    a = _make(analyzer_cls, CURRENT)
    a.client.messages.response = _Response([_Block("text", text="{}")])
    a._call_api("hello")
    assert a.client.messages.calls[0]["output_config"] == {"effort": "low"}


def test_request_marks_the_system_prompt_as_cacheable(analyzer_cls):
    """The stable prefix carries the breakpoint; article text follows it."""
    a = _make(analyzer_cls, CURRENT)
    a.client.messages.response = _Response([_Block("text", text="{}")])
    a._call_api("hello")
    system = a.client.messages.calls[0]["system"]
    assert isinstance(system, list)
    assert system[0]["cache_control"] == {"type": "ephemeral"}


def test_request_sends_no_deprecated_thinking_budget(analyzer_cls):
    """budget_tokens is rejected outright on the current models."""
    a = _make(analyzer_cls, CURRENT)
    a.client.messages.response = _Response([_Block("text", text="{}")])
    a._call_api("hello")
    call = a.client.messages.calls[0]
    assert "thinking" not in call or "budget_tokens" not in (
        call.get("thinking") or {}
    )


def test_request_does_not_force_tool_choice(analyzer_cls):
    """Forced tool_choice is a 400 on the current models."""
    a = _make(analyzer_cls, CURRENT)
    a.client.messages.response = _Response([_Block("text", text="{}")])
    a._call_api("hello")
    assert "tool_choice" not in a.client.messages.calls[0]


# ─────────────────────────────────────────────────────────────────────
# Cost accounting
# ─────────────────────────────────────────────────────────────────────

def test_cost_uses_the_selected_models_rate(analyzer_cls):
    a = _make(analyzer_cls, CURRENT, model="claude-sonnet-5-5")
    a.total_input_tokens = 1_000_000
    a.total_output_tokens = 1_000_000
    stats = a.get_usage_stats()
    # Sonnet 5.5: $2 in, $10 out per MTok.
    assert stats["estimated_cost_usd"] == pytest.approx(12.0)
    assert stats["model"] == "claude-sonnet-5-5"


def test_cost_differs_between_tiers(analyzer_cls):
    cheap = _make(analyzer_cls, CURRENT, model="claude-haiku-5-5")
    dear = _make(analyzer_cls, CURRENT, model="claude-opus-5-5")
    for a in (cheap, dear):
        a.total_input_tokens = 1_000_000
        a.total_output_tokens = 1_000_000
    assert (cheap.get_usage_stats()["estimated_cost_usd"]
            < dear.get_usage_stats()["estimated_cost_usd"])


def test_dated_snapshot_prices_like_its_base_model(analyzer_cls):
    a = _make(analyzer_cls, CURRENT, model="claude-sonnet-5-5-20260401")
    assert a._rates_per_mtok() == (2.00, 10.00)


def test_unknown_model_still_reports_a_cost(analyzer_cls):
    a = _make(analyzer_cls, CURRENT, model="claude-something-new")
    a.total_input_tokens = 1_000_000
    assert a.get_usage_stats()["estimated_cost_usd"] > 0


def test_usage_accumulates_across_calls(analyzer_cls):
    a = _make(analyzer_cls, CURRENT)
    a.client.messages.response = _Response(
        [_Block("text", text="{}")], usage=_Usage(10, 5),
    )
    a._call_api("one")
    a._call_api("two")
    assert a.total_calls == 2
    assert a.total_input_tokens == 20
    assert a.total_output_tokens == 10
