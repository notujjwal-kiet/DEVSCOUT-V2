"""
Shared Pydantic v2 domain models for DEVSCOUT Learning Companion.

Every pipeline stage takes and returns types defined here.
Strict Pydantic v2 models: no untyped dicts or dict[str, Any].
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import MAX_EXCERPT_CHARS


# ── Identifiers & Timestamps ──────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(UTC)


def _new_id() -> str:
    return str(uuid4())


# ── Enums ──────────────────────────────────────────────────────────────────────

class ExperienceLevel(StrEnum):
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"


class ResourceType(StrEnum):
    DOCS = "docs"
    REPO = "repo"
    BLOG = "blog"
    COMMUNITY = "community"
    PAPER = "paper"


class QuestionType(StrEnum):
    CONCEPTUAL = "conceptual"
    EXERCISE = "exercise"
    EXPLAIN_BACK = "explain_back"


class CodeGateLevel(StrEnum):
    EXPLAIN = "EXPLAIN"
    SNIPPET = "SNIPPET"
    FULL = "FULL"
    REQUIRE_CHECK = "REQUIRE_CHECK"


class AttemptStatus(StrEnum):
    ACCEPTABLE = "ACCEPTABLE"
    TRIVIAL = "TRIVIAL"
    COPIED_SNIPPET = "COPIED_SNIPPET"
    INVALID = "INVALID"


class PipelineStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ── Stage Telemetry ────────────────────────────────────────────────────────────

class StageMetrics(BaseModel):
    """Latency, token, and cost tracking for a single pipeline stage."""

    stage_id: int = Field(..., ge=1, le=9, description="Pipeline stage number (1-9)")
    stage_name: str
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime | None = None
    latency_ms: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0


# ── Stage 1 – Developer Goal & Skill Context ──────────────────────────────────

class LearnerProfile(BaseModel):
    """Developer skill level, stack experience, and current project constraints."""

    learner_id: str = Field(default_factory=_new_id)
    experience_level: ExperienceLevel = ExperienceLevel.BEGINNER
    tech_stack: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    current_project: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)


class LearningGoal(BaseModel):
    """What the developer wants to build and learn."""

    goal_id: str = Field(default_factory=_new_id)
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=10, max_length=2000)
    target_skills: list[str] = Field(default_factory=list)
    project_context: str | None = None
    constraints: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


# ── Stage 2 – Intent & Learning Plan ──────────────────────────────────────────

class Milestone(BaseModel):
    """Concrete building milestone within a learning plan."""

    milestone_id: str = Field(default_factory=_new_id)
    title: str = Field(..., min_length=3, max_length=200)
    description: str = Field(..., min_length=5, max_length=1000)
    target_outcome: str = Field(..., min_length=3, max_length=1000)
    estimated_duration_minutes: int | None = Field(default=None, ge=1)
    is_completed: bool = False


class LearningPlan(BaseModel):
    """Structured breakdown: knowledge gaps, outcomes, depth, milestones."""

    plan_id: str = Field(default_factory=_new_id)
    goal_id: str
    knowledge_gaps: list[str] = Field(default_factory=list)
    learning_outcomes: list[str] = Field(default_factory=list)
    depth_level: str = "intermediate"
    milestones: list[Milestone] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


# ── Stage 3 – Research & Source Discovery ─────────────────────────────────────

class Resource(BaseModel):
    """Discovered real source with provenance, freshness, and fitness."""

    model_config = ConfigDict(validate_assignment=True)

    resource_id: str = Field(default_factory=_new_id)
    url: str
    type: ResourceType
    title: str
    retrieved_at: datetime = Field(default_factory=_utcnow)
    content_hash: str
    reliability: float = Field(default=0.0, ge=0.0, le=1.0)
    freshness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    level_fit: ExperienceLevel = ExperienceLevel.BEGINNER
    uncertainty_flag: bool = False
    summary: str = ""
    excerpt: str = Field(default="", exclude=True)
    last_updated: datetime | None = None
    popularity: int | None = None
    topic: str | None = None

    @field_validator("excerpt")
    @classmethod
    def _bound_excerpt(cls, v: str) -> str:
        if v and len(v) > MAX_EXCERPT_CHARS:
            return v[:MAX_EXCERPT_CHARS]
        return v

    @property
    def evidence_score(self) -> float:
        return self.reliability * self.freshness_score


# ── Stage 5 – Explanation & Concept Mapping ───────────────────────────────────

class ExplanationSection(BaseModel):
    """Section of an explanation with citations to verified Resources."""

    title: str
    content: str
    citation_resource_ids: list[str] = Field(default_factory=list)
    weakly_supported: bool = False


class Explanation(BaseModel):
    """Conceptual explanation backed by real sources and examples."""

    explanation_id: str = Field(default_factory=_new_id)
    topic: str
    summary: str
    sections: list[ExplanationSection] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    concept_relationships: list[str] = Field(default_factory=list)
    real_world_examples: list[str] = Field(default_factory=list)
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)


class ConceptNode(BaseModel):
    """Node in the conceptual knowledge map."""

    id: str
    label: str
    description: str
    difficulty: ExperienceLevel = ExperienceLevel.BEGINNER


class ConceptEdge(BaseModel):
    """Directed relationship between concepts (e.g. prerequisite, builds_on)."""

    source: str
    target: str
    relation: str


# ── Code Snippets & Gate Policy ───────────────────────────────────────────────

class Snippet(BaseModel):
    """Short code snippet permitted under strict pedagogical code policy."""

    snippet_id: str = Field(default_factory=_new_id)
    code: str
    language: str = "python"
    concept: str
    line_count: int = Field(..., ge=1, le=15)
    explanation: str = Field(..., min_length=5)

    @model_validator(mode="after")
    def validate_line_count(self) -> Snippet:
        actual_lines = len(self.code.strip().splitlines())
        if actual_lines > 15:
            raise ValueError(f"Snippet exceeds max 15 lines (has {actual_lines} lines)")
        return self


class CodeGateDecision(BaseModel):
    """Deterministic output from Code Assistance Gate."""

    decision_id: str = Field(default_factory=_new_id)
    level: CodeGateLevel
    reasons: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)


class AttemptReview(BaseModel):
    """Deterministic review of a learner's code attempt."""

    status: AttemptStatus
    reasons: list[str] = Field(default_factory=list)


# ── Stage 7 – Understanding Check ─────────────────────────────────────────────

class UnderstandingQuestion(BaseModel):
    """Conceptual question, small exercise, or explain-back prompt."""

    question_id: str = Field(default_factory=_new_id)
    concept_id: str
    question_text: str
    question_type: QuestionType = QuestionType.CONCEPTUAL
    expected_points: list[str] = Field(default_factory=list)


class UnderstandingResult(BaseModel):
    """Outcome of evaluating learner's answer or attempt."""

    result_id: str = Field(default_factory=_new_id)
    question_id: str
    passed: bool
    learner_response: str
    feedback: str
    detected_misconceptions: list[str] = Field(default_factory=list)
    score: float = Field(default=1.0, ge=0.0, le=1.0)


# ── Stage 9 – Progress & Reflection ───────────────────────────────────────────

class ProgressRecord(BaseModel):
    """Persisted learning history, reflections, and next steps."""

    record_id: str = Field(default_factory=_new_id)
    learner_id: str
    goal_id: str
    completed_topics: list[str] = Field(default_factory=list)
    reflection_notes: str | None = None
    next_recommended_steps: list[str] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=_utcnow)


# ── Unified Pipeline Response ─────────────────────────────────────────────────

class LearningResponse(BaseModel):
    """Complete response returned to developer after pipeline validation."""

    response_id: str = Field(default_factory=_new_id)
    learner_profile: LearnerProfile
    learning_plan: LearningPlan | None = None
    resources: list[Resource] = Field(default_factory=list)
    explanation: Explanation | None = None
    snippets: list[Snippet] = Field(default_factory=list)
    full_code: str | None = None
    concept_nodes: list[ConceptNode] = Field(default_factory=list)
    concept_edges: list[ConceptEdge] = Field(default_factory=list)
    questions: list[UnderstandingQuestion] = Field(default_factory=list)
    understanding_result: UnderstandingResult | None = None
    code_gate_decision: CodeGateDecision
    progress: ProgressRecord | None = None
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    metrics: list[StageMetrics] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)


class LearnRequest(BaseModel):
    """Input payload for initiating a learning session via POST /api/learn."""

    goal: str = Field(..., min_length=3, max_length=2000, description="What the developer wants to learn or build")
    project: str | None = Field(default=None, description="Current project context or name")
    experience_level: ExperienceLevel = Field(default=ExperienceLevel.BEGINNER)
    tech_stack: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    target_skills: list[str] = Field(default_factory=list)
    wants_snippet: bool = Field(default=False)


class AnswerSubmission(BaseModel):
    """A learner's answer to an understanding question."""

    question_id: str
    answer: str


class CodeRequest(BaseModel):
    """Input payload for requesting code or submitting answers via POST /api/code-request."""

    response_id: str
    wants_full_code: bool = True
    attempt_code: str | None = None
    attempt_language: str | None = None
    answers: list[AnswerSubmission] | None = None


class FullCodeSolution(BaseModel):
    """Structured LLM output for verified full code release."""

    code: str
    language: str = "python"
    explanation: str
    line_by_line_explanation: list[str] = Field(default_factory=list)



# ── Stage Typed Return Models (for stages 1-9) ────────────────────────────────

class GoalContextResult(BaseModel):
    profile: LearnerProfile
    goal: LearningGoal
    metrics: StageMetrics | None = None


class IntentPlanResult(BaseModel):
    plan: LearningPlan
    metrics: StageMetrics | None = None


class SourceDiscoveryResult(BaseModel):
    goal_id: str
    resources: list[Resource] = Field(default_factory=list)
    loop_count: int = 0
    metrics: StageMetrics | None = None


class EvidenceReviewResult(BaseModel):
    goal_id: str
    is_sufficient: bool
    is_fresh: bool
    reviewed_resources: list[Resource] = Field(default_factory=list)
    flagged_uncertainties: list[str] = Field(default_factory=list)
    needs_more_evidence: bool = False
    loop_count: int = 0
    seen_urls: list[str] = Field(default_factory=list)
    metrics: StageMetrics | None = None

    @property
    def exclude_urls(self) -> list[str]:
        """Convenience alias for orchestrator to pass seen_urls to Stage 3 exclude_urls."""
        return self.seen_urls


class ExplanationResult(BaseModel):
    explanation: Explanation
    concept_nodes: list[ConceptNode] = Field(default_factory=list)
    concept_edges: list[ConceptEdge] = Field(default_factory=list)
    metrics: StageMetrics | None = None


class GuidedWorkStep(BaseModel):
    step_id: str = Field(default_factory=_new_id)
    step_number: int
    title: str
    guidance: str
    hints: list[str] = Field(default_factory=list)
    learner_task: str


class GuidedWorkResult(BaseModel):
    goal_id: str
    steps: list[GuidedWorkStep] = Field(default_factory=list)
    metrics: StageMetrics | None = None


class UnderstandingCheckResult(BaseModel):
    questions: list[UnderstandingQuestion] = Field(default_factory=list)
    results: list[UnderstandingResult] = Field(default_factory=list)
    metrics: StageMetrics | None = None


class CodeGateResult(BaseModel):
    decision: CodeGateDecision
    allowed_snippets: list[Snippet] = Field(default_factory=list)
    full_code: str | None = None
    metrics: StageMetrics | None = None


class ProgressReflectionResult(BaseModel):
    progress: ProgressRecord
    metrics: StageMetrics | None = None
