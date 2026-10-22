# DEVSCOUT — Evidence-Backed Developer Learning Companion

DEVSCOUT is an AI-powered **learning companion** for software engineers building real-world projects. Instead of acting as an automated code generator or a copy-paste oracle, DEVSCOUT guides developers through hands-on implementation with source-grounded conceptual explanations, sequential milestone planning, comprehension checks, and an evidence-based code assistance gate.

The backend exposes a typed FastAPI service structured around a 9-stage learning pipeline, deterministic code gate policies, and strict source-quote matching.

---

## What DEVSCOUT Is (and What It Isn't)

- **A Learning Companion, NOT a Code Generator**: Passive code generation damages developer retention and understanding. DEVSCOUT never presents ungrounded model outputs as authoritative facts.
- **Checked Against the Source Text**: Every explanation claim must link directly to authoritative technical resources (official documentation, engineering repositories, specifications). Supporting quotes must match the fetched source text verbatim.
- **Gated Code Assistance**: Full code solutions are locked behind an evidence-based gate. The learner receives conceptual direction and bounded snippets first, unlocking full implementations only after demonstrating understanding or submitting an earnest attempt.

---

## The 9-Stage Learning Pipeline

```
[1 Goal Context] ──► [2 Learning Plan] ──► [3 Source Discovery] ◄──┐
                             ▲                     │                │ (Evidence loop:
                             │                     ▼                │  max retry loops)
                             │             [4 Evidence Review] ─────┘
                             │                     │
                             │                     ▼
                             │       [5 Concept Mapping & Explanation]
                             │                     │
                             │                     ▼
                             │           [6 Guided Project Work]
                             │                     │
                             │                     ▼
                             │          [7 Understanding Check]
                             │                     │
                             │                     ▼
                             │         [8 Code Assistance Gate]
                             │                     │
                             │                     ▼
                             └──────── [9 Progress & Reflection]
```

| Stage | Name | Module | Primary Responsibility |
|:---:|:---|:---|:---|
| **1** | **Developer Goal & Skill Context** | `app/pipeline/stage1_developer_input.py` | Captures learner profile, technical goal, experience level (beginner/intermediate/advanced), tech stack, and constraints. |
| **2** | **Intent & Learning Plan** | `app/pipeline/stage2_input_processing.py` | Structures learning outcomes, knowledge gaps, appropriate depth, and sequential milestones. |
| **3** | **Research & Source Discovery** | `app/pipeline/stage3_research.py` | Discovers multi-perspective sources (official docs, repos, community guides) via Gemini grounded search or seed docs fallback; fetches text safely against SSRF. |
| **4** | **Evidence Review** | `app/pipeline/stage4_evidence_validation.py` | Evaluates domain authority, time-decay freshness, and tech-stack relevance; flags conflicting or malicious content. Triggers retry loop (4 ➔ 3) if sources are insufficient. |
| **5** | **Explanation & Concept Mapping** | `app/pipeline/stage5_ai_analysis.py` | Generates deep pedagogical explanations and a concept ontology graph, strictly anchoring claims to excerpts checked against the source text. |
| **6** | **Guided Project Work** | `app/pipeline/stage6_recommendation.py` | Produces step-by-step implementation tasks, debugging hints, and self-guided checkpoints without dumping full solutions. |
| **7** | **Understanding Check** | `app/pipeline/stage7_output_validation.py` | Generates 2–3 targeted comprehension questions (conceptual, explain-back, exercise). Evaluates learner answers using a deterministic heuristic. |
| **8** | **Code Assistance Gate** | `app/pipeline/stage8_decision_report.py` | Enforces the deterministic code policy, deciding whether `EXPLAIN`, `SNIPPET`, `REQUIRE_CHECK`, or `FULL` code may be returned. |
| **9** | **Progress & Reflection** | `app/pipeline/stage9_developer_feedback.py` | Records learner milestones, reflections, telemetry metrics, and prepares context updates for future sessions. |

### Feedback & Retry Loops
- **Evidence Loop (Stage 4 ➔ Stage 3)**: When discovered sources lack official documentation or fail freshness thresholds, Stage 4 triggers a targeted retry to Stage 3 with seen URLs excluded (capped by `max_evidence_retry_loops`).
- **Seed Documentation Fallback**: If grounded web search fails or runs out of quota, authoritative curated seed documentation (FastAPI, Python, Pydantic, React, TypeScript) is automatically ingested.

---

## Code Policy and the Hint Ladder

The code assistance gate (`app/policy/code_gate.py`) is entirely deterministic and **LLM-free**. It governs what code assistance can be exposed based on learner readiness:

```
┌─────────────────────────────────────────────────────────────┐
│                       THE HINT LADDER                       │
├─────────────────┬───────────────────────────────────────────┤
│ 1. EXPLAIN      │ Default level. Conceptual guidance,       │
│                 │ trade-offs, and architecture. Zero code.  │
├─────────────────┼───────────────────────────────────────────┤
│ 2. SNIPPET      │ Maximum 15 lines of code. Covers exactly   │
│                 │ one concept. Non-runnable. Max 3 snippets.│
├─────────────────┼───────────────────────────────────────────┤
│ 3. REQUIRE_CHECK│ Returned when full code is requested but  │
│                 │ learner readiness is not yet unlocked.    │
├─────────────────┼───────────────────────────────────────────┤
│ 4. FULL         │ Complete, runnable implementation with    │
│                 │ line-by-line pedagogical breakdown.       │
└─────────────────┴───────────────────────────────────────────┘
```

- **`EXPLAIN`**: Default response. No code blocks are permitted; only architectural explanations and concept relationships.
- **`SNIPPET`**: Allowed when requested or deemed pedagogically beneficial. Strict bounds:
  - At most **15 lines** of code.
  - Demonstrates exactly **one concept**.
  - Never a complete runnable application or boilerplate entrypoint (no `if __name__ == '__main__':` or full server wiring).
  - Maximum 3 snippets per milestone response.
- **`REQUIRE_CHECK`**: If full code is requested before readiness criteria are met, the gate rejects the request and returns `REQUIRE_CHECK` alongside comprehension questions.
- **`FULL`**: Released only after readiness has been unlocked. Full code is always accompanied by line-by-line explanations.

---

## How Readiness Is Decided

Learner readiness is tracked per session in `app/policy/code_gate.py` and `app/policy/attempt_review.py`. Full code solutions are unlocked through any of three explicit paths:

1. **Understanding Check Passed**:
   - The learner submits answers to the understanding questions generated in Stage 7 (`POST /api/learn/answers`).
   - The learner must correctly answer at least `min_questions_answered` (default 2).
   - **Deterministic Heuristic**: Answer evaluation does **not** rely on an arbitrary AI judge. It uses a deterministic keyword and token overlap heuristic, calculating whether the learner's answer covers at least 60% of the key concept points without requiring exact phrasing.
2. **Reviewed Code Attempt**:
   - The learner submits their own code attempt (`POST /api/learn/code/attempt`).
   - Must contain substantive effort: at least **10 non-blank, non-comment lines of code**.
   - Must parse cleanly as valid code (e.g. valid Python AST).
   - Must reference core concept terms from the milestone.
   - **Anti-Plagiarism Protection**: Normalized token similarity against DEVSCOUT's previously provided hints and snippets must be **under 80%** (the learner cannot copy and paste assistant hints back).
3. **Final Build Submission**:
   - The learner flags that their project milestone implementation is complete and submits it for final review.

---

## How Citations Are Checked & Confidence Scoring

### Verbatim Citation Checking
In `app/pipeline/response_validator.py`, every claim in an explanation section must cite an existing resource ID and include a `supporting_quote`:
- **Verbatim Text Match**: The quote **must appear verbatim in the fetched source text** (the scraped document or authoritative excerpt).
- **Zero Hallucinated Citations**: If a quote does not appear in the source text, or cites a non-existent resource ID, the citation is dropped during deterministic response repair.
- **Unsupported Status**: Any section claim that cannot be anchored directly to retrieved documentation is explicitly marked as `support_status = "unsupported"` and forbidden from displaying phantom citations.

### Confidence Score Composition
The overall `confidence_score` (between `0.0` and `1.0`) is computed deterministically from:
1. **Source Authority**: Presence of official documentation and primary technical repositories.
2. **Domain Diversity**: Requirement of at least two distinct authoritative domains.
3. **Source Type Diversity**: Combination of official guides, code repositories, and technical discussions.
4. **Time-Decay Freshness**: Penalties applied to outdated materials.
5. **Quote Grounding Ratio**: Proportion of sections backed by verbatim quotes checked against the source text.

If the confidence score falls below `0.65`, a `LOW_CONFIDENCE` or `NO_RELIABLE_SOURCES` warning is attached to the response.

---

## Quota, Model Registry, and Routing

DEVSCOUT incorporates a multi-tier model management system designed to operate reliably within strict API rate limits (e.g., Gemini free tier: 4 RPM, 18 requests/day).

### 1. Model Registry (`models.json`)
Models are configured in `models.json` and validated at startup using Pydantic:
- **Supported Providers**: `gemini`, `openai_compatible`.
- **Security Rule**: `base_url` is operator-controlled configuration and must never come from a user request. HTTPS is enforced for remote endpoints (HTTP permitted solely for localhost endpoints).

### 2. Per-Model Rate Gate (`app/llm/rate_gate.py`)
- **Sliding-Window RPM Limiter**: Every model has an independent sliding window (default 4 RPM for Gemini) that queues requests asynchronously via `asyncio.sleep` rather than failing.
- **Daily Budget Limit (RPD)**: Daily request counter persisted in `data/quota.json` (resets at midnight in UTC-8). If the daily budget is exhausted, it raises `LLMProviderError("Gemini daily budget reached")` immediately before dispatching the request.
- **Local Models**: Local/self-hosted models are designated as unlimited.

### 3. Dynamic Router & Fallback (`app/llm/router.py`)
- Evaluates candidate models by priority and available daily budget.
- If a primary model encounters rate limits or daily quota exhaustion, the router automatically fails over to any configured secondary model.
- Grounded web search is restricted to Gemini models with active budget.

### 4. File-Based Caches
- **Request Cache (`data/responses.json`)**: Normalizes request parameters (`goal`, `tech_stack`, `experience_level`) into a SHA-256 key, returning cached plans instantly on repeated requests.
- **Research Cache (`data/research_cache.json`)**: Caches search queries and scraped web documents by normalized URL with a 24-hour TTL.

---

## Setup and Running

### Prerequisites
- Python 3.11 or 3.12
- Google Gemini API key

### 1. Clone & Create Virtual Environment

```bash
git clone https://github.com/notujjwal-kiet/DEVSCOUT-V2.git
cd DEVSCOUT-V2

# Create virtual environment
python -m venv .venv

# Activate on Windows:
.\.venv\Scripts\Activate.ps1

# Activate on macOS / Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment (`.env`)

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

Set your configuration values:

```env
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000

# Primary LLM provider
LLM_PROVIDER=gemini

# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
GEMINI_RPM=4
GEMINI_RPD_LIMIT=18
```

### 3. Start the Backend Server

```bash
python -m uvicorn app.main:app --reload --port 8000
```

The interactive API documentation is accessible at `http://localhost:8000/docs`.

### 4. Running the Diagnostic Script

To verify that model generation, rate gating, and grounded discovery work properly in your environment:

```bash
python scripts/diag.py
```

### 5. Frontend Setup

The companion UI repository is available at:
👉 **[DEVSCOUT-V2 Frontend Repository](https://github.com/notujjwal-kiet/DEVSCOUT-V2-FRONTEND)**

In the frontend directory, set up `.env`:

```env
VITE_API_BASE_URL=http://localhost:8000
```

Install packages and run:

```bash
npm install
npm run dev
```

---

## Running Tests

The test suite contains 247 comprehensive tests covering pipeline stages, code gate enforcement, citation checks, rate limiting, and model routing. Every test executes against temporary paths with zero modifications to the project's `data/` directory and zero live network calls.

```bash
# Run full suite
pytest

# Run with duration profiling
pytest --durations=15
```

The entire suite executes in approximately **5.5 seconds**.

---

## Honest Limitations

1. **In-Memory Session & Job Store**: Active learning sessions, async jobs, and question state reside in server memory (with automatic TTL eviction and a 1,000-session cap). Restarting the backend server drops in-flight interactive sessions (persisted disk caches and quotas remain intact).
2. **Deterministic Heuristic Answer Checking**: Understanding question evaluation uses keyword and token overlap heuristics rather than subjective AI grading or human review.
3. **Free-Tier Gemini Quotas**: Default limits of 4 RPM and 18 requests per day require requests to queue asynchronously when limits are reached.
4. **No Multi-User Authentication**: Intended for individual developer local setups or controlled private deployments. Bearer token protection acts as a shared API key rather than full user account isolation.

---

## Roadmap

- [ ] **Persistent Database Layer**: SQLite and PostgreSQL adapters to preserve learner session state, milestone notes, and reflections across server restarts.
- [ ] **Interactive Socratic Dialogue**: Multi-turn milestone conversation interface for asking clarifying questions while coding.
- [ ] **Sandboxed Code Execution**: Docker / WebAssembly runner to execute and validate unit tests written by learners directly inside the companion UI.
- [ ] **Expanded Local Models**: Fine-tuned small language models for zero-quota, completely offline technical mentoring.
