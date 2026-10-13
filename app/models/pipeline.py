"""
Shared Pydantic domain models.

Every pipeline stage takes and returns types defined here (or types that
compose these).  No untyped dicts are passed between stages.
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Identifiers ────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return str(uuid4())


# ── Enums ──────────────────────────────────────────────────────────────────────

class PipelineStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ResearchMode(StrEnum):
    LIVE = "live"
    CACHED = "cached"


# ── Stage telemetry ────────────────────────────────────────────────────────────

class StageMetrics(BaseModel):
    """Latency, token, and cost tracking for a single pipeline stage."""

    stage_id: int = Field(..., ge=1, le=9, description="Pipeline stage number (1-9)")
    stage_name: str
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime | None = None
    latency_ms: float | None = None          # populated on completion
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0                    # always 0 for local Ollama


# ── Stage 1 – Developer Input ──────────────────────────────────────────────────

class DeveloperInput(BaseModel):
    """Raw request from the developer / frontend."""

    request_id: str = Field(default_factory=_new_id)
    question: str = Field(..., min_length=10, max_length=2000)
    context: str | None = Field(default=None, max_length=4000)
    created_at: datetime = Field(default_factory=_utcnow)


# ── Stage 2 – Input Processing ─────────────────────────────────────────────────

class ProcessedInput(BaseModel):
    """Structured, normalised representation of the developer question."""

    request_id: str
    original_question: str
    normalized_question: str
    entities: list[str] = Field(default_factory=list)   # e.g. ["PostgreSQL", "MongoDB"]
    intent: str | None = None                             # e.g. "database_selection"
    context: str | None = None
    metrics: StageMetrics | None = None


# ── Stage 3 – Research & Evidence Collection ───────────────────────────────────

class EvidenceItem(BaseModel):
    """A single piece of collected evidence."""

    evidence_id: str = Field(default_factory=_new_id)
    source_url: str
    title: str
    snippet: str
    fetched_at: datetime = Field(default_factory=_utcnow)
    is_cached: bool = False


class ResearchResult(BaseModel):
    """All evidence gathered for a given processed input."""

    request_id: str
    evidence_items: list[EvidenceItem] = Field(default_factory=list)
    research_mode: ResearchMode = ResearchMode.CACHED
    loop_count: int = 0
    metrics: StageMetrics | None = None


# ── Stage 4 – Source Freshness & Evidence Validation ──────────────────────────

class ValidationResult(BaseModel):
    """Outcome of evidence freshness / sufficiency check."""

    request_id: str
    is_sufficient: bool
    is_fresh: bool
    issues: list[str] = Field(default_factory=list)
    loop_count: int = 0
    validated_evidence: list[EvidenceItem] = Field(default_factory=list)
    metrics: StageMetrics | None = None


# ── Stage 5 – AI Decision Analysis ────────────────────────────────────────────

class AnalysisResult(BaseModel):
    """LLM-generated analysis of the evidence."""

    request_id: str
    analysis_text: str
    trade_offs: dict[str, Any] = Field(default_factory=dict)
    key_factors: list[str] = Field(default_factory=list)
    raw_llm_response: str | None = None
    metrics: StageMetrics | None = None


# ── Stage 6 – Recommendation Generation ───────────────────────────────────────

class Recommendation(BaseModel):
    """A single ranked recommendation."""

    rank: int
    option: str
    rationale: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    citations: list[str] = Field(default_factory=list)   # evidence_id references


class RecommendationResult(BaseModel):
    """Full set of recommendations for the question."""

    request_id: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    overall_confidence: float = Field(..., ge=0.0, le=1.0)
    metrics: StageMetrics | None = None


# ── Stage 7 – Output Validation ────────────────────────────────────────────────

class OutputValidationResult(BaseModel):
    """Schema, citation, and confidence validation outcome."""

    request_id: str
    is_valid: bool
    schema_valid: bool = False
    citations_valid: bool = False
    confidence_acceptable: bool = False
    repair_attempted: bool = False
    issues: list[str] = Field(default_factory=list)
    metrics: StageMetrics | None = None


# ── Stage 8 – Decision Report ──────────────────────────────────────────────────

class DecisionReport(BaseModel):
    """Final structured report returned to the frontend."""

    report_id: str = Field(default_factory=_new_id)
    request_id: str
    question: str
    summary: str
    recommendations: list[Recommendation]
    overall_confidence: float
    evidence_items: list[EvidenceItem]
    generated_at: datetime = Field(default_factory=_utcnow)
    pipeline_metrics: list[StageMetrics] = Field(default_factory=list)


# ── Stage 9 – Developer Feedback ──────────────────────────────────────────────

class DeveloperFeedback(BaseModel):
    """Feedback submitted by the developer after reviewing a report."""

    feedback_id: str = Field(default_factory=_new_id)
    report_id: str
    request_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: str | None = None
    was_helpful: bool | None = None
    submitted_at: datetime = Field(default_factory=_utcnow)


class FeedbackAck(BaseModel):
    """Acknowledgement returned after storing feedback."""

    feedback_id: str
    stored: bool
    message: str
