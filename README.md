# DEVSCOUT — AI-Powered Technical Decision Intelligence Platform

Local-first AI-powered platform that answers technical decision questions
(e.g. *"PostgreSQL vs MongoDB for my project?"*) with evidence-backed
reports, trade-off analysis, source citations, and confidence scores.

The **backend** (this repo) exposes a REST API consumed by a separate
React + TypeScript frontend.

---

## Architecture Overview

```
Frontend (React/TS)  ──►  FastAPI Backend  ──►  Strands Agent (Ollama LLM)
                                │
                         PostgreSQL + pgvector
                         (evidence store & memory)
```

---

## The 9-Stage Pipeline

Every developer question flows through these stages in order.
Stage names are used verbatim in code, logs, and metrics.

| # | Stage | Module | Responsibility |
|---|-------|--------|----------------|
| 1 | **Developer Input** | `stage1_developer_input` | Accept & validate the raw question; assign `request_id` |
| 2 | **Input Processing** | `stage2_input_processing` | Normalise text, extract named entities, classify intent |
| 3 | **Research & Evidence Collection** | `stage3_research` | Web search (live/cached) + pgvector similarity recall |
| 4 | **Source Freshness & Evidence Validation** | `stage4_evidence_validation` | Check recency & sufficiency; loop back to Stage 3 (max 2×) if insufficient |
| 5 | **AI Decision Analysis** | `stage5_ai_analysis` | LLM (Ollama via Strands) generates trade-offs & key factors |
| 6 | **Recommendation Generation** | `stage6_recommendation` | Rank options, attach confidence scores and citation IDs |
| 7 | **Output Validation** | `stage7_output_validation` | Schema check, citation cross-ref, confidence threshold; repair if invalid |
| 8 | **Decision Report** | `stage8_decision_report` | Assemble & return the final structured report |
| 9 | **Developer Feedback** | `stage9_developer_feedback` | Store rating/comment; update Context & Memory layer |

> Stage 4 loops back to Stage 3 up to **`MAX_EVIDENCE_RETRY_LOOPS`** times
> (default 2) before proceeding with what it has.

---

## Project Structure

```
DEVSCOUT-V2/
├── app/
│   ├── main.py              # FastAPI app factory
│   ├── config.py            # Pydantic-settings config loader
│   ├── api/
│   │   ├── router.py        # Aggregates all route modules
│   │   └── health.py        # GET /health
│   ├── pipeline/
│   │   ├── stage1_developer_input.py
│   │   ├── stage2_input_processing.py
│   │   ├── stage3_research.py
│   │   ├── stage4_evidence_validation.py
│   │   ├── stage5_ai_analysis.py
│   │   ├── stage6_recommendation.py
│   │   ├── stage7_output_validation.py
│   │   ├── stage8_decision_report.py
│   │   └── stage9_developer_feedback.py
│   ├── agents/
│   │   └── base_agent.py    # Strands Agent wrapper (placeholder)
│   ├── db/
│   │   └── session.py       # Async SQLAlchemy engine (placeholder)
│   └── models/
│       └── pipeline.py      # All shared Pydantic domain models
├── tests/
│   ├── conftest.py
│   ├── test_config.py
│   ├── test_health.py
│   ├── test_models.py
│   └── test_pipeline_stages.py
├── docs/
│   └── sql/
│       └── init.sql         # pgvector + uuid-ossp bootstrap
├── .env.example
├── docker-compose.yml       # PostgreSQL 16 + pgvector
├── pytest.ini
├── requirements.txt
├── run.py                   # uvicorn entry point
└── README.md
```

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| Docker + Compose | v2+ |
| Ollama | latest (with at least one model pulled) |

---

## Quick Start

### 1 — Clone & create virtual environment

```bash
git clone <repo-url>
cd DEVSCOUT-V2

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2 — Configure environment

```bash
cp .env.example .env
# Edit .env – set OLLAMA_MODEL to a model you have pulled locally,
# e.g. OLLAMA_MODEL=llama3.2
```

### 3 — Start PostgreSQL (with pgvector)

```bash
docker compose up -d postgres
# Verify it is healthy:
docker compose ps
```

### 4 — Run the API

```bash
python run.py
# Or directly:
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 5 — Check it works

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "ok",
  "version": "0.1.0",
  "environment": "development",
  "ollama_host": "http://localhost:11434",
  "ollama_model": "llama3.2",
  "timestamp": "2026-10-13T16:03:01.000000+00:00"
}
```

Interactive API docs (development only): <http://localhost:8000/docs>

---

## Running Tests

```bash
pytest -v
```

---

## Environment Variables Reference

See [`.env.example`](.env.example) for the full list with descriptions.

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `OLLAMA_MODEL` | `llama3.2` | Model name (must be pulled locally) |
| `DATABASE_URL` | `postgresql+asyncpg://…` | Async PostgreSQL DSN |
| `WEB_RESEARCH_MODE` | `cached` | `live` or `cached` evidence fetching |
| `MAX_EVIDENCE_RETRY_LOOPS` | `2` | Stage 4 → Stage 3 max retry loops |
| `CONFIDENCE_THRESHOLD` | `0.65` | Minimum acceptable confidence score |

AWS variables are **optional** and gated behind an interface — the core pipeline never imports them.

---

## What Is Done / Not Done

### ✅ Done (Scaffold)
- FastAPI app with `/health` endpoint
- Pydantic-settings config loader (all values from env)
- 9 pipeline stage placeholder modules (typed I/O, structured logging, metrics stubs)
- Shared Pydantic domain models for all stage inputs/outputs (`StageMetrics`, `DeveloperInput` … `FeedbackAck`)
- Docker Compose for PostgreSQL 16 + pgvector
- `pytest` suite: config, health, models, and all 9 stage placeholders
- `.env.example`, `requirements.txt`, `.gitignore`

### 🔲 Not Done (Next Tasks)
- Stage 2: NLP entity extraction via Strands agent
- Stage 3: Live web search + pgvector similarity recall
- Stage 4: Freshness heuristics, domain trust scoring
- Stage 5: Full Strands Agent + Ollama integration, prompt templates, token tracking
- Stage 6: Ranking algorithm, citation linking, evidence-based confidence scoring
- Stage 7: Citation cross-referencing, auto-repair prompts
- Stage 8: PostgreSQL persistence of reports
- Stage 9: DB feedback storage, context & memory embedding updates
- API endpoints for `/analyze`, `/report/{id}`, `/feedback`
- Alembic migrations
- AWS abstraction layer (optional)
