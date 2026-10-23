# DEVSCOUT

### Learn While You Build.

DEVSCOUT is an AI-assisted developer learning platform that guides learners through project planning, research, concept explanations, guided implementation, understanding checks, and readiness-based code assistance.

This repository contains the FastAPI backend. The React + TypeScript frontend is maintained at [Devscout-UI](https://github.com/NotUjjwal-95/Devscout-UI).

> **Status note:** On the inspected main branch, the only registered API route is `GET /health`. All other endpoints and provider integrations below are targets until implemented and tested. Update this README as the code changes.

---

## Product Principles

- **Teach concepts and support learning, not just generate code.**
- **Distinguish research-backed claims from model-generated explanations.**
- **Keep provider credentials on the backend.**
- **Enforce code-readiness checks in deterministic backend logic, never through a prompt-only rule.**
- **Keep mock/demo responses distinct from live provider output.**
- **Describe only verified features as implemented.**

---

## Intended Learning Journey

1. **Developer goal and skill context**
2. **Intent classification and learning plan**
3. **Research and source discovery**
4. **Evidence review**
5. **Concept explanation and mapping**
6. **Guided project work**
7. **Understanding check**
8. **Code assistance gate**
9. **Progress and reflection**

*These are intended product stages, not a claim that all nine are currently integrated.*

---

## Intended Architecture

```text
React + TypeScript frontend
           │
           │  HTTP / JSON
           ▼
    FastAPI backend
    ├── Request validation and API routes
    ├── Task-based model/provider routing
    ├── Research and learning workflow
    └── Deterministic code-readiness policy
           │
           ├── OpenAI-compatible hosted chat providers (Groq)
           ├── Fallback hosted chat provider (Cerebras)
           ├── Google Gemini for selected research tasks
           ├── Optional local Ollama model
           └── PostgreSQL + pgvector where required
```

*The browser calls the backend; it must never call model providers directly with private credentials.*

---

## Provider Strategy

The planned provider layer uses a shared OpenAI-compatible client only for providers that support the relevant API format:

- **Groq**: candidate primary hosted chat provider.
- **Cerebras**: candidate fallback, depending on account access and current quotas.
- **Google AI Studio / Gemini**: selected research tasks to conserve usage.
- **Ollama**: optional local inference fallback.

Provider availability, model IDs, free-tier quotas, and supported parameters vary. OpenAI-compatible does not mean providers share quotas or support every identical feature. Use bounded retries, respect `Retry-After`, and do not repeatedly retry exhausted quotas, authentication failures, or invalid requests.

---

## API Status on Inspected Main

| Method and Route | Status on Inspected Main | Current Implementation Status (`learning-pivot`) |
| :--- | :--- | :--- |
| `GET /health` | **Implemented** | **Implemented** (checks Gemini & uptime) |
| `GET /api/models` | Not registered | **Implemented & Verified** (Groq, Cerebras, Gemini, Ollama) |
| `POST /api/chat` | Not registered | **Implemented & Verified** (multi-provider with failover) |
| `POST /api/learn` | Not registered | **Implemented & Verified** (background job dispatch) |
| `GET /api/learn/{job_id}` | Not registered | **Implemented & Verified** (job status & results polling) |
| `POST /api/code-request` | Not registered | **Implemented & Verified** (deterministic code gate) |
| `GET /api/quota` | Not registered | **Implemented & Verified** (rate gate usage tracking) |

*Keep this table aligned with the actual router and verified tests. Never use fake success responses to conceal unfinished routes.*

---

## Requirements

- Python 3.11+
- Git
- Optional: Docker Desktop / Docker Compose for database services
- Optional: Ollama with a downloaded model
- API keys for whichever hosted providers you actually configure

Python dependencies are listed in [`requirements.txt`](requirements.txt).

---

## Run Locally

### 1. Clone and Install

```bash
git clone https://github.com/notujjwal-kiet/DEVSCOUT-V2.git
cd DEVSCOUT-V2
python -m venv .venv
```

**Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

**macOS/Linux:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

### 2. Configure Environment

Edit `.env` for the features you intend to run. Example provider variable names for the planned integration:

```env
GROQ_API_KEY=
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.3-70b-versatile
CEREBRAS_API_KEY=
CEREBRAS_BASE_URL=https://api.cerebras.ai/v1
CEREBRAS_MODEL=llama3.1-8b
GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.5-flash
OLLAMA_ENABLED=false
OLLAMA_BASE_URL=http://localhost:11434/v1
OLLAMA_MODEL=llama3.2
```

These variables have no effect until the application reads them. Never commit `.env`, hardcode credentials, log keys, or put provider secrets in frontend `VITE_*` variables.

### 3. Start the API

```bash
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) or [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- Interactive API documentation: available at `/docs` in development if enabled by app configuration.

### 4. Run Tests

```bash
pytest -v
```

*Report tests as passing only after running them in the current environment.*

---

## Priorities

1. Implement and test provider registry and bounded fallback.
2. Align registered API endpoints with the frontend contract.
3. Complete one end-to-end learning journey with real source attribution.
4. Enforce the code-readiness gate server-side.
5. Add usage/quota visibility and persistence where required by the demo.

---

## Security and Reliability

- Keep provider keys server-side and out of logs.
- Validate requests and use bounded network timeouts.
- Configure CORS for the intended deployment origins.
- Treat free-tier limits and provider availability as variable.
- Separate mock data from live responses.
- Do not describe placeholders or untested paths as complete.

---

## Related Repository

Frontend: [NotUjjwal-95/Devscout-UI](https://github.com/NotUjjwal-95/Devscout-UI)
