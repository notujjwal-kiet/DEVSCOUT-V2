"""
tests/test_token_telemetry.py

Test Suite 5: Token & Cost Telemetry
- Fake Gemini client with usage_metadata -> tokens, latency, and cost recorded
- Local response with tokens recorded -> cost 0.0 for local models
- Provider without usage -> tokens None (not 0)
- Unknown model -> cost None (null)
Zero network calls: mock response objects and pure telemetry methods.
"""
from __future__ import annotations

from unittest.mock import MagicMock
import pytest
from pydantic import BaseModel

from app.config import calculate_token_cost, get_settings
from app.models.pipeline import StageMetrics


class _FakeUsageMetadata:
    def __init__(self, prompt_tokens: int, candidates_tokens: int, total_tokens: int):
        self.prompt_token_count = prompt_tokens
        self.candidates_token_count = candidates_tokens
        self.total_token_count = total_tokens


class _FakeGeminiResponse:
    def __init__(self, prompt_tokens: int = 1200, completion_tokens: int = 400):
        self.usage_metadata = _FakeUsageMetadata(
            prompt_tokens=prompt_tokens,
            candidates_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )
        self.text = '{"dummy": "data"}'


def test_gemini_usage_metadata_tokens_latency_and_cost():
    """Gemini response with usage_metadata records prompt, completion, total tokens and calculated cost."""
    gemini_resp = _FakeGeminiResponse(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    metrics = StageMetrics(
        stage_id=5,
        stage_name="Explanation & Concept Mapping",
        latency_ms=245.5,
    )

    metrics.fill_tokens(gemini_resp, model_name="gemini-2.5-flash")

    assert metrics.prompt_tokens == 1_000_000
    assert metrics.completion_tokens == 1_000_000
    assert metrics.total_tokens == 2_000_000
    assert metrics.latency_ms == 245.5

    # Pricing for gemini-2.5-flash: $0.075 / 1M input, $0.30 / 1M output -> Total $0.375
    assert metrics.cost_usd is not None
    assert round(metrics.cost_usd, 4) == 0.3750


def test_local_model_recorded_zero_cost():
    """Local response with prompt_tokens and completion_tokens records tokens and sets cost to 0.0."""
    resp_dict = {
        "prompt_tokens": 850,
        "completion_tokens": 350,
        "total_tokens": 1200,
    }
    metrics = StageMetrics(
        stage_id=2,
        stage_name="Intent & Learning Plan",
        latency_ms=180.0,
    )

    metrics.fill_tokens(resp_dict, model_name="local-model", is_local=True)

    assert metrics.prompt_tokens == 850
    assert metrics.completion_tokens == 350
    assert metrics.total_tokens == 1200
    assert metrics.latency_ms == 180.0
    # Local model is local, so cost must be 0.0
    assert metrics.cost_usd == 0.0


def test_provider_without_usage_leaves_tokens_none():
    """Provider response without usage metadata leaves token fields as None, not 0."""
    class _EmptyResponse:
        pass

    empty_resp = _EmptyResponse()
    metrics = StageMetrics(
        stage_id=1,
        stage_name="Developer Goal Context",
        latency_ms=15.0,
    )

    metrics.fill_tokens(empty_resp, model_name="gemini-2.5-flash")

    # Fields must be None, NOT 0
    assert metrics.prompt_tokens is None
    assert metrics.completion_tokens is None
    assert metrics.total_tokens is None
    assert metrics.cost_usd is None


def test_unknown_model_leaves_cost_none():
    """Known tokens but unknown model in pricing table sets cost to None (null)."""
    gemini_resp = _FakeGeminiResponse(prompt_tokens=500, completion_tokens=200)
    metrics = StageMetrics(
        stage_id=5,
        stage_name="Explanation",
    )

    metrics.fill_tokens(gemini_resp, model_name="unregistered-custom-model-v99")

    assert metrics.prompt_tokens == 500
    assert metrics.completion_tokens == 200
    assert metrics.total_tokens == 700
    # Cost cannot be calculated for unknown models -> None
    assert metrics.cost_usd is None


def test_calculate_token_cost_function_direct():
    """Unit tests for config.calculate_token_cost pricing lookup."""
    # Local model is always 0.0
    assert calculate_token_cost("local-model", 1000, 500, is_local=True) == 0.0

    # Both None tokens -> None cost
    assert calculate_token_cost("gemini-2.5-flash", None, None) is None

    # Unknown model -> None cost
    assert calculate_token_cost("nonexistent-model-xyz", 1000, 1000) is None

    # Known Flash rates (0.075 and 0.30 per 1M)
    flash_cost = calculate_token_cost("gemini-2.5-flash", 1_000_000, 1_000_000)
    assert flash_cost == 0.375

    # Known Pro rates (1.25 and 5.0 per 1M)
    pro_cost = calculate_token_cost("gemini-1.5-pro", 1_000_000, 1_000_000)
    assert pro_cost == 6.25
