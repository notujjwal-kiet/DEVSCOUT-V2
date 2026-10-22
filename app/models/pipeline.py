"""
Shared Pydantic v2 domain models for DEVSCOUT Learning Companion.

Every pipeline stage takes and returns types defined here.
Strict Pydantic v2 models: no untyped dicts or dict[str, Any].
"""
from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.config import get_settings


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
    IRRELEVANT = "IRRELEVANT"


class PipelineStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


# ── Stage Telemetry ────────────────────────────────────────────────────────────

class StageMetrics(BaseModel):
    """Latency, token, and cost tracking for a single pipeline stage."""

    stage_id: int = Field(..., ge=1, le=9, description="Pipeline stage number (1-9)")
    stage_name: str
    started_at: datetime = Field(default_factory=_utcnow)
    ended_at: datetime | None = None
    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    model_id: str | None = None

    def fill_tokens(
        self,
        response_or_provider: object,
        model_name: str | None = None,
        is_ollama: bool = False,
    ) -> None:
        """
        Fill token counts and cost from provider response or provider state:
        - Gemini: usage_metadata (prompt_token_count, candidates_token_count, total_token_count)
        - Ollama: prompt_eval_count, eval_count
        - If no usage: prompt_tokens, completion_tokens, total_tokens remain None (not 0)
        - Cost: computed from config price table; 0.0 for Ollama, None for unknown models.
        """
        from app.config import calculate_token_cost, get_settings

        if hasattr(response_or_provider, "model_id"):
            mid = getattr(response_or_provider, "model_id", None)
            if isinstance(mid, str) and mid:
                self.model_id = mid

        prompt_val: int | None = None
        cand_val: int | None = None
        total_val: int | None = None
        detected_ollama = is_ollama

        # 1. Check for Ollama-style usage fields directly on response/dict or provider
        ollama_data = None
        if isinstance(response_or_provider, dict) and (
            "prompt_eval_count" in response_or_provider or "eval_count" in response_or_provider
        ):
            ollama_data = response_or_provider
        elif hasattr(response_or_provider, "last_usage_data") and isinstance(
            getattr(response_or_provider, "last_usage_data", None), dict
        ):
            ollama_data = getattr(response_or_provider, "last_usage_data")

        if ollama_data is not None:
            detected_ollama = True
            p = ollama_data.get("prompt_eval_count")
            c = ollama_data.get("eval_count")
            if p is not None:
                prompt_val = int(p)
            if c is not None:
                cand_val = int(c)
            if prompt_val is not None or cand_val is not None:
                total_val = (prompt_val or 0) + (cand_val or 0)

        # 2. Check for Gemini-style usage_metadata
        if prompt_val is None and cand_val is None:
            usage = getattr(response_or_provider, "last_usage_metadata", None)
            if usage is None:
                usage = getattr(response_or_provider, "usage_metadata", None)
            if usage is None and isinstance(response_or_provider, dict):
                usage = response_or_provider.get("usage_metadata")

            if usage is not None:
                if isinstance(usage, dict):
                    p = usage.get("prompt_token_count")
                    c = usage.get("candidates_token_count")
                    t = usage.get("total_token_count")
                else:
                    p = getattr(usage, "prompt_token_count", None)
                    c = getattr(usage, "candidates_token_count", None)
                    t = getattr(usage, "total_token_count", None)

                if p is not None:
                    prompt_val = int(p)
                if c is not None:
                    cand_val = int(c)
                if t is not None:
                    total_val = int(t)
                elif prompt_val is not None or cand_val is not None:
                    total_val = (prompt_val or 0) + (cand_val or 0)

        # Assign tokens (remain None if not found)
        self.prompt_tokens = prompt_val
        self.completion_tokens = cand_val
        self.total_tokens = total_val

        # 3. Model name resolution
        resolved_model = model_name
        if resolved_model is None:
            resolved_model = getattr(response_or_provider, "model_name", None) or getattr(
                response_or_provider, "model", None
            )
        if resolved_model is None and not detected_ollama:
            cfg = get_settings()
            resolved_model = cfg.gemini_model or "gemini-2.5-flash"

        if resolved_model and "ollama" in str(resolved_model).lower():
            detected_ollama = True

        # 4. Compute cost
        if prompt_val is None and cand_val is None:
            self.cost_usd = 0.0 if detected_ollama else None
        else:
            self.cost_usd = calculate_token_cost(
                model_name=resolved_model,
                prompt_tokens=self.prompt_tokens,
                completion_tokens=self.completion_tokens,
                is_ollama=detected_ollama,
            )



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
    concepts: list[str] = Field(default_factory=list)
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
    fetched_text: str = Field(default="", exclude=True)
    origin: str = Field(default="discovered")
    is_rejected: bool = False
    rejection_reason: str | None = None

    @field_validator("excerpt")
    @classmethod
    def _bound_excerpt(cls, v: str) -> str:
        max_chars = get_settings().max_excerpt_chars
        if v and len(v) > max_chars:
            return v[:max_chars]
        return v

    @property
    def evidence_score(self) -> float:
        return self.reliability * self.freshness_score


# ── Stage 5 – Explanation & Concept Mapping ───────────────────────────────────

class Citation(BaseModel):
    """Citation linking an explanation section to a resource with supporting quote."""

    resource_id: str
    supporting_quote: str = Field(default="", max_length=1000)


class ExplanationSection(BaseModel):
    """Section of an explanation with citations to verified Resources."""

    title: str
    content: str
    citations: list[Citation] = Field(default_factory=list)
    citation_resource_ids: list[str] = Field(default_factory=list)
    support_status: Literal["supported", "weakly_supported", "unsupported"] = "supported"
    weakly_supported: bool = False

    @model_validator(mode="before")
    @classmethod
    def _sync_citations_and_ids(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        cits = data.get("citations")
        rids = data.get("citation_resource_ids")
        if cits and not rids:
            extracted_ids = []
            for c in cits:
                if isinstance(c, dict):
                    extracted_ids.append(c.get("resource_id", ""))
                elif hasattr(c, "resource_id"):
                    extracted_ids.append(getattr(c, "resource_id"))
            data["citation_resource_ids"] = [str(i) for i in extracted_ids if i]
        elif rids and not cits:
            data["citations"] = [
                {"resource_id": str(rid), "supporting_quote": ""}
                for rid in rids
                if rid
            ]

        status = data.get("support_status")
        weakly = data.get("weakly_supported")
        if status == "weakly_supported":
            data["weakly_supported"] = True
        elif status == "unsupported":
            data["weakly_supported"] = False
        elif weakly:
            data["support_status"] = "weakly_supported"
        return data


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
    unlocked_by: str | None = None


class AttemptReview(BaseModel):
    """Deterministic review of a learner's code attempt."""

    status: AttemptStatus
    reasons: list[str] = Field(default_factory=list)


# ── Stage 7 – Understanding Check ─────────────────────────────────────────────

class PublicUnderstandingQuestion(BaseModel):
    """Public-facing question model: no expected_points exposed to clients."""

    question_id: str = Field(default_factory=_new_id)
    concept_id: str
    question_text: str
    question_type: QuestionType = QuestionType.CONCEPTUAL


class UnderstandingQuestion(PublicUnderstandingQuestion):
    """Internal question model: includes expected_points (answer key)."""

    expected_points: list[str] = Field(default_factory=list)

    def to_public(self) -> PublicUnderstandingQuestion:
        return PublicUnderstandingQuestion(
            question_id=self.question_id,
            concept_id=self.concept_id,
            question_text=self.question_text,
            question_type=self.question_type,
        )


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

class ResponseWarning(BaseModel):
    """Warning surfaced to the learner regarding sources or claims."""

    severity: Literal["info", "warning", "critical"]
    code: str
    message: str


class RejectedSource(BaseModel):
    url: str
    reason: str


class EvidenceSummary(BaseModel):
    discovered: int = 0
    fetched: int = 0
    accepted: int = 0
    rejected: list[RejectedSource] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


class LearningResponse(BaseModel):
    """Complete response returned to developer after pipeline validation."""

    response_id: str = Field(default_factory=_new_id)
    learner_profile: LearnerProfile
    learning_plan: LearningPlan | None = None
    resources: list[Resource] = Field(default_factory=list)
    explanation: Explanation | None = None
    snippets: list[Snippet] = Field(default_factory=list)
    full_code: str | None = None
    code_notice: str | None = None
    concept_nodes: list[ConceptNode] = Field(default_factory=list)
    concept_edges: list[ConceptEdge] = Field(default_factory=list)
    questions: list[PublicUnderstandingQuestion] = Field(default_factory=list)
    understanding_result: UnderstandingResult | None = None
    code_gate_decision: CodeGateDecision
    unlocked_by: str | None = None
    progress: ProgressRecord | None = None
    confidence_score: float = Field(default=1.0, ge=0.0, le=1.0)
    quality_score: float = Field(default=1.0, ge=0.0, le=1.0)
    evidence_status: str = Field(default="sufficient")
    evidence_summary: EvidenceSummary = Field(default_factory=EvidenceSummary)
    warnings: list[ResponseWarning] = Field(default_factory=list)
    metrics: list[StageMetrics] = Field(default_factory=list)
    models_used: list[str] = Field(default_factory=list, description="Ordered, unique list of models used across stages")
    from_cache: bool = False
    generated_at: datetime | None = None
    created_at: datetime = Field(default_factory=_utcnow)

    @field_validator("questions", mode="before")
    @classmethod
    def _coerce_public_questions(cls, v: object) -> list[PublicUnderstandingQuestion]:
        if not v or not isinstance(v, (list, tuple)):
            return []
        res: list[PublicUnderstandingQuestion] = []
        for item in v:
            if isinstance(item, PublicUnderstandingQuestion) and type(item) is PublicUnderstandingQuestion:
                res.append(item)
            elif hasattr(item, "question_id") and hasattr(item, "concept_id") and hasattr(item, "question_text"):
                res.append(
                    PublicUnderstandingQuestion(
                        question_id=item.question_id,
                        concept_id=item.concept_id,
                        question_text=item.question_text,
                        question_type=getattr(item, "question_type", QuestionType.CONCEPTUAL),
                    )
                )
            elif isinstance(item, dict):
                res.append(
                    PublicUnderstandingQuestion(
                        question_id=item.get("question_id") or _new_id(),
                        concept_id=item.get("concept_id", ""),
                        question_text=item.get("question_text", ""),
                        question_type=item.get("question_type", QuestionType.CONCEPTUAL),
                    )
                )
            else:
                res.append(item)
        return res

    def to_public(self) -> "LearningResponse":
        """Return a copy with expected_points stripped from questions."""
        import copy
        clone = copy.deepcopy(self)
        clone.questions = [
            PublicUnderstandingQuestion(
                question_id=q.question_id,
                concept_id=q.concept_id,
                question_text=q.question_text,
                question_type=q.question_type,
            )
            for q in clone.questions
        ]
        return clone


class LearnRequest(BaseModel):
    """Input payload for initiating a learning session via POST /api/learn."""

    goal: str = Field(..., min_length=3, max_length=1000, description="What the developer wants to learn or build")
    description: str | None = Field(default=None, max_length=2000, description="Optional detailed description")
    project: str | None = Field(default=None, description="Current project context or name")
    experience_level: ExperienceLevel = Field(default=ExperienceLevel.BEGINNER)
    tech_stack: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    target_skills: list[str] = Field(default_factory=list)
    wants_snippet: bool = Field(default=False)
    refresh: bool = Field(default=False, description="Bypass request-level response cache if true")
    model_id: str | None = Field(default=None, description="Optional requested model ID from models.json registry")

    @field_validator("model_id")
    @classmethod
    def _validate_model_id(cls, v: str | None) -> str | None:
        if v is None:
            return None
        from app.llm.registry import get_model_registry

        reg = get_model_registry()
        if not reg.has_model(v):
            raise ValueError(
                f"Unknown model_id '{v}'. Available models: {reg.get_model_ids()}"
            )
        return v


class LearnJobRecord(BaseModel):
    """Record tracking background execution of a learning pipeline job."""

    job_id: str = Field(default_factory=_new_id)
    status: JobStatus = Field(default=JobStatus.PENDING)
    progress: float = Field(default=0.0, ge=0.0, le=1.0)
    current_stage: str | None = None
    stage_id: int | None = None
    result: LearningResponse | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)



class AnswerSubmission(BaseModel):
    """A learner's answer to an understanding question."""

    question_id: str
    answer: str


class CodeRequest(BaseModel):
    """Input payload for requesting code or submitting answers via POST /api/code-request."""

    response_id: str
    wants_full_code: bool = True
    attempt_code: str | None = Field(default=None, max_length=50000)
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
    discovered_count: int = 0
    fetched_count: int = 0
    errors: list[str] = Field(default_factory=list)
    loop_count: int = 0
    degraded: bool = False
    rate_limited_searches: int = 0
    metrics: StageMetrics | None = None


class EvidenceReviewResult(BaseModel):
    goal_id: str
    is_sufficient: bool
    is_fresh: bool
    reviewed_resources: list[Resource] = Field(default_factory=list)
    rejected_resources: list[Resource] = Field(default_factory=list)
    flagged_uncertainties: list[str] = Field(default_factory=list)
    needs_more_evidence: bool = False
    loop_count: int = 0
    seen_urls: list[str] = Field(default_factory=list)
    insufficient_topics: list[str] = Field(default_factory=list)
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
