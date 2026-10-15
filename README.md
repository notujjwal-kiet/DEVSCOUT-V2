# DEVSCOUT — Learning Companion ("Learn While You Build")

DEVSCOUT is an AI-powered **learning companion** for developers building real projects. Instead of acting as an oracle that spits out copy-paste code, DEVSCOUT guides developers through hands-on implementation with source-backed conceptual explanations, incremental task breakdowns, understanding checks, and a strict code assistance gate.

The **backend** (this repo) exposes a typed FastAPI service designed around a 9-stage pipeline, deterministic code gate policies, and cross-cutting response verification.

---

## Core Principles

1. **AI is a guide, not the authority**: The system never presents its own assertions as ground truth.
2. **Explanations backed by real sources**: Every conceptual claim must cite verified resources (docs, repositories, engineering blogs, research papers). Never treat a claim as verified merely because an AI model agrees.
3. **Learning over copy-paste**: Passive code consumption harms retention. Implementation remains the developer's responsibility; code assistance is strictly gated by verified understanding.

---

## The 9-Stage Pipeline

```
[1 Goal & Context] ──► [2 Learning Plan] ──► [3 Source Discovery] ◄──┐
                              ▲                      │               │ (4->3 loop:
                              │                      ▼               │  max 2x)
                              │              [4 Evidence Review] ────┘
                              │                      │
                              │                      ▼
                              │         [5 Concept Mapping & Explanation]
                              │                      │
                              │                      ▼
                              │            [6 Guided Project Work]
                              │                      │
                              │                      ▼
                              │           [7 Understanding Check]
                              │                      │
                              │                      ▼
                              │          [8 Code Assistance Gate]
                              │                      │
                              │                      ▼
                              └───────── [9 Progress & Reflection]
```

| # | Stage | Module | Responsibility |
|---|-------|--------|----------------|
| 1 | **Developer Goal & Skill Context** | `stage1_developer_input` | Capture project goal, developer experience level, tech stack, and constraints. |
| 2 | **Intent & Learning Plan** | `stage2_input_processing` | Identify knowledge gaps, targeted outcomes, required depth, and sequential milestones. |
| 3 | **Research & Source Discovery** | `stage3_research` | Discover multi-perspective resources across official docs, GitHub repos, engineering blogs, and community discussions. |
| 4 | **Evidence Review** | `stage4_evidence_validation` | Check provenance, freshness via time-decay, relevance, flag conflicting/uncertain claims, and hash content. |
| 5 | **Explanation & Concept Mapping** | `stage5_ai_analysis` | Provide clear conceptual explanations citing verified resources, map concept graphs, prerequisites, and real-world examples. |
| 6 | **Guided Project Work** | `stage6_recommendation` | Break down project into manageable, task-by-task guidance with debugging hints while the learner implements. |
| 7 | **Understanding Check** | `stage7_output_validation` | Test comprehension with conceptual questions, small exercises, explain-back prompts, and misconception detection. |
| 8 | **Code Assistance Gate** | `stage8_decision_report` | Deterministic policy gate deciding what code level, if any, may be released. |
| 9 | **Progress & Reflection** | `stage9_developer_feedback` | Persist learning history, completed topics, reflections, and next steps into context. |

### Feedback Loops
- **Evidence Loop (4 ➔ 3)**: If evidence is insufficient or stale, stage 4 loops back to stage 3 for additional discovery (max 2 loops).
- **Inquiry Loop (User questions ➔ 2/3)**: Clarifying questions dynamically update the learning plan or trigger targeted source discovery.
- **Reflection Loop (9 ➔ 2)**: Progress and feedback persist into context, updating future milestones and plans.

---

## Code Policy (`app/policy/code_gate.py`)

Deterministic and **LLM-free**. Governed by `evaluate_code_gate`:

- **Default**: `EXPLAIN`. No code is returned; only conceptual guidance.
- **`SNIPPET` allowed when useful**:
  - Maximum **15 lines** of code.
  - Covers exactly **one concept**.
  - Never a complete runnable file or full wiring (e.g., no entrypoints like `if __name__ == '__main__':` or full app setups).
  - Maximum 3 snippets per response.
- **`FULL` implementation only if**:
  1. `explicit_request` is `True`, **AND**
  2. `readiness` criteria are met.
- **Readiness Rule**:
  - Configurable centrally in `ReadinessPolicyConfig`.
  - Satisfied if: **understanding check passed** OR **learner submitted their own attempt** OR **final build submitted**.
- **Requirement on unmet readiness**:
  - If full code is requested but readiness is not met, the gate returns `REQUIRE_CHECK`, never code.
- **All returned code must come with a pedagogical explanation**.

```python
CodeGateDecision(
    level=CodeGateLevel.EXPLAIN | SNIPPET | FULL | REQUIRE_CHECK,
    reasons=list[str],
    violations=list[str],
)
```

---

## Cross-Cutting Validator (`app/pipeline/response_validator.py`)

Executed on every outgoing response to guarantee integrity:

1. **Schema Valid**: Verifies strict Pydantic v2 `LearningResponse` domain structure with zero untyped dictionaries.
2. **Citation Integrity**: Every explanation section claim **must cite a verified Resource ID** that exists in the response. Nonexistent resource IDs or missing citations fail validation.
3. **Code Gate Enforcement**: Scans all explanation text and markdown fenced code blocks (```` ``` ````) to ensure no unauthorized code blocks violate `EXPLAIN` or `REQUIRE_CHECK` gates, and that snippets do not exceed 15 lines.
4. **Quality & Confidence**: Verifies `confidence_score` and `quality_score` are present in `[0.0, 1.0]`.
5. **Deterministic Repair Hook**: `repair_response` automatically sanitizes violations (strips unauthorized code blocks, filters bad citation IDs, clamps scores) without calling an LLM.

---

## LLM Provider Interface (`app/llm/provider.py`)

A decoupled `LLMProvider` Protocol with stub implementations (no external network calls during testing):
- **OllamaProvider**: Local inference stub with configurable model (`OLLAMA_MODEL`, default `llama3.2`).
- **GeminiProvider**: Cloud model stub with configurable model (`GEMINI_MODEL`, default `gemini-1.5-flash`).
- Provider selection controlled via `LLM_PROVIDER` environment variable (`ollama` | `gemini`).

---

## Project Structure

```
DEVSCOUT-V2/
├── app/
│   ├── main.py                     # FastAPI application factory
│   ├── config.py                   # Pydantic-settings configuration loader
│   ├── api/
│   │   ├── router.py               # Aggregates API routes
│   │   └── health.py               # GET /health
│   ├── llm/
│   │   ├── __init__.py
│   │   └── provider.py             # LLMProvider Protocol + Ollama/Gemini stubs
│   ├── policy/
│   │   ├── __init__.py
│   │   └── code_gate.py            # Deterministic code assistance gate & readiness config
│   ├── pipeline/
│   │   ├── __init__.py             # Pipeline exports & semantic aliases
│   │   ├── response_validator.py   # Cross-cutting validator and repair hook
│   │   ├── stage1_developer_input.py      # Goal & Skill Context
│   │   ├── stage2_input_processing.py     # Intent & Learning Plan
│   │   ├── stage3_research.py             # Research & Source Discovery
│   │   ├── stage4_evidence_validation.py  # Evidence Review (4->3 loop)
│   │   ├── stage5_ai_analysis.py          # Explanation & Concept Mapping
│   │   ├── stage6_recommendation.py       # Guided Project Work
│   │   ├── stage7_output_validation.py    # Understanding Check
│   │   ├── stage8_decision_report.py      # Code Assistance Gate evaluation
│   │   └── stage9_developer_feedback.py   # Progress & Reflection
│   ├── models/
│   │   ├── __init__.py
│   │   └── pipeline.py             # Strict Pydantic v2 schemas (no dict[str, Any])
│   ├── db/
│   │   └── session.py              # Async SQLAlchemy session setup
│   └── agents/
│       └── base_agent.py           # Agent scaffold
├── tests/
│   ├── test_code_gate.py           # Thorough tests for code gate policy
│   ├── test_response_validator.py  # Citation, gate violation, and repair tests
│   ├── test_llm_provider.py        # LLM protocol, stubs, and env selection
│   ├── test_models.py              # Domain model validations and bounds
│   ├── test_pipeline_stages.py     # Integration tests for all 9 stages
│   ├── test_config.py              # Configuration and environment tests
│   └── test_health.py              # Health check endpoint tests
├── pytest.ini
├── requirements.txt
└── README.md
```

---

## Running Tests

```bash
pytest -v
```

---

## Done / Not-Done List

### Done
- [x] **Pivoted Architecture**: Refactored backend from decision report tool to "Learn while you build" Learning Companion.
- [x] **Pydantic v2 Domain Models**: Rewrote `app/models/pipeline.py` with zero `dict[str, Any]`, strictly typing `LearnerProfile`, `ExperienceLevel`, `LearningGoal`, `LearningPlan`, `Milestone`, `Resource`, `Explanation`, `Snippet`, `ConceptNode`, `ConceptEdge`, `UnderstandingQuestion`, `UnderstandingResult`, `CodeGateDecision`, `ProgressRecord`, `LearningResponse`, and `StageMetrics`.
- [x] **Deterministic Code Assistance Gate**: Built `app/policy/code_gate.py` with default `EXPLAIN`, 15-line single-concept snippet validation, max 3 snippets, and configurable readiness rule requiring an understanding check, learner attempt, or final build.
- [x] **Cross-Cutting Response Validator**: Built `app/pipeline/response_validator.py` enforcing schema compliance, valid resource citation IDs on every claim, fenced code block scanning against gate decisions, quality metrics, and deterministic repair hook.
- [x] **LLM Provider Abstraction**: Added `app/llm/provider.py` with `LLMProvider` Protocol, `OllamaProvider` and `GeminiProvider` stubs (no network calls), selected by `LLM_PROVIDER` with configurable model names.
- [x] **9 Pipeline Stages Updated**: Updated signatures, docstrings, and typed models for stages 1 through 9.
- [x] **Thorough Test Suite**: Added 45 tests across `test_code_gate.py`, `test_response_validator.py`, `test_llm_provider.py`, `test_models.py`, `test_pipeline_stages.py`, `test_config.py`, and `test_health.py`.

### Not Done (Future Work)
- [ ] Live web search integrations (e.g. Tavily, SearXNG, GitHub Code Search) in Stage 3.
- [ ] Vector database indexing with PostgreSQL + pgvector for embedding-based resource retrieval.
- [ ] LLM agent prompts with live inference for personalized explanation generation and dynamic misconception detection.
- [ ] Interactive frontend UI (React + TypeScript) communicating over REST with code gate visualizations and quiz exercises.
- [ ] Database persistence for learner profiles and progress history.
