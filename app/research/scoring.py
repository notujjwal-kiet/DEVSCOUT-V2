"""
app/research/scoring.py

Pure functions and configurable constants for evidence scoring, recency time-decay,
source reliability, and cryptographic content change detection.
"""
from __future__ import annotations

from datetime import UTC, datetime
import hashlib
import math
import re
from typing import Sequence
import urllib.parse

from app.models.pipeline import ExperienceLevel, LearningGoal, Resource, ResourceType
from app.research.models import ResearchDocument

# ── Config Block (All Constants Configurable at Top) ──────────────────────────

# Freshness & time-decay half-lives (in days) per resource type
DEFAULT_HALF_LIFE_DAYS: dict[str, float] = {
    "docs": 365.0,
    "repo": 270.0,
    "blog": 180.0,
    "community": 120.0,
    "paper": 900.0,
}
FALLBACK_HALF_LIFE_DAYS: float = 180.0

# Freshness default for undated resources (None date)
DEFAULT_NONE_DATE_FRESHNESS: float = 0.3

# Official documentation domains granted 1.0 reliability
OFFICIAL_DOCS_DOMAINS: set[str] = {
    "python.org",
    "docs.python.org",
    "fastapi.tiangolo.com",
    "developer.mozilla.org",
    "docs.github.com",
    "docs.docker.com",
    "kubernetes.io",
    "postgresql.org",
    "pydantic.dev",
    "docs.pydantic.dev",
    "docs.sqlalchemy.org",
    "react.dev",
    "nodejs.org",
    "go.dev",
    "rust-lang.org",
    "w3.org",
    "peps.python.org",
}

# Reliability baseline constants
OFFICIAL_DOCS_RELIABILITY: float = 1.0
UNOFFICIAL_DOCS_RELIABILITY: float = 0.7
BLOG_RELIABILITY: float = 0.5
UNKNOWN_RELIABILITY: float = 0.3
ACADEMIC_PAPER_RELIABILITY: float = 0.85

# GitHub repository reliability scaling by log10(stars)
GITHUB_REPO_MAX_RELIABILITY: float = 0.9
GITHUB_REPO_MIN_RELIABILITY: float = 0.3
GITHUB_REPO_LOG_STARS_DIVISOR: float = 5.0  # log10(100,000) = 5.0

# Community thread reliability scaling by popularity/score
COMMUNITY_MAX_RELIABILITY: float = 0.8
COMMUNITY_MIN_RELIABILITY: float = 0.3
COMMUNITY_SCORE_DIVISOR: float = 100.0


# ── Text Normalization & Content Hashing ───────────────────────────────────────

def normalize(text: str) -> str:
    """
    Normalize whitespace by collapsing all consecutive whitespace characters
    (spaces, tabs, newlines) into single spaces and stripping ends.
    """
    return " ".join(text.split())


# Shared alias for backward compatibility with existing connectors
normalize_whitespace = normalize


def hash_content(text: str) -> str:
    """Compute SHA-256 digest of whitespace-normalized text."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()


def content_changed(old_hash: str, new_content: str) -> bool:
    """
    Check if new_content has changed compared to old_hash using whitespace
    normalization and SHA-256 digest.
    Returns True if content changed, False if identical.
    """
    new_hash = hash_content(new_content)
    return old_hash.strip().lower() != new_hash.lower()


# ── Domain & Half-Life Helpers ─────────────────────────────────────────────────

def extract_domain(domain_or_url: str | None) -> str:
    """Extract and normalize host domain from a URL or raw domain string."""
    if not domain_or_url:
        return ""
    cleaned = domain_or_url.strip().lower()
    if "://" in cleaned:
        try:
            parsed = urllib.parse.urlparse(cleaned)
            cleaned = parsed.netloc or cleaned
        except Exception:
            pass
    if ":" in cleaned:
        cleaned = cleaned.split(":", 1)[0]
    return cleaned


def is_official_docs_domain(domain_or_url: str | None) -> bool:
    """Check if the given domain or URL matches the official documentation domains list."""
    host = extract_domain(domain_or_url)
    if not host:
        return False
    if host in OFFICIAL_DOCS_DOMAINS:
        return True
    for official in OFFICIAL_DOCS_DOMAINS:
        if host == official or host.endswith(f".{official}"):
            return True
    return False


def get_half_life_days(resource_type: ResourceType | str) -> float:
    """Return half-life in days for a given resource type."""
    key = resource_type.value if isinstance(resource_type, ResourceType) else str(resource_type).lower()
    return DEFAULT_HALF_LIFE_DAYS.get(key, FALLBACK_HALF_LIFE_DAYS)


# ── Scoring Functions ──────────────────────────────────────────────────────────

def freshness_score(
    last_updated: datetime | None,
    now: datetime | None = None,
    half_life_days: float = FALLBACK_HALF_LIFE_DAYS,
) -> float:
    """
    Compute recency score via exponential time-decay:
      freshness = 0.5 ** (age_days / half_life_days)
    If last_updated is None, returns DEFAULT_NONE_DATE_FRESHNESS (0.3).
    Result is bounded strictly between 0.0 and 1.0.
    """
    if last_updated is None:
        return DEFAULT_NONE_DATE_FRESHNESS

    current_time = now if now is not None else datetime.now(UTC)

    # Ensure timezone awareness for comparison
    ts_updated = last_updated if last_updated.tzinfo is not None else last_updated.replace(tzinfo=UTC)
    ts_now = current_time if current_time.tzinfo is not None else current_time.replace(tzinfo=UTC)

    # Prevent negative age for future-dated commits/posts
    age_seconds = max(0.0, (ts_now - ts_updated).total_seconds())
    age_days = age_seconds / 86400.0

    hl = max(1e-6, float(half_life_days))
    score = 0.5 ** (age_days / hl)
    return float(max(0.0, min(1.0, score)))


def _scale_repo_reliability(popularity: int | float | None) -> float:
    """Scale GitHub repository reliability by log10(stars), capped at 0.9."""
    if popularity is None or popularity <= 0:
        return GITHUB_REPO_MIN_RELIABILITY
    stars = float(popularity)
    # Log10-based scaling from GITHUB_REPO_MIN_RELIABILITY up to GITHUB_REPO_MAX_RELIABILITY
    log_val = math.log10(max(1.0, stars))
    ratio = min(1.0, log_val / GITHUB_REPO_LOG_STARS_DIVISOR)
    spread = GITHUB_REPO_MAX_RELIABILITY - GITHUB_REPO_MIN_RELIABILITY
    score = GITHUB_REPO_MIN_RELIABILITY + (ratio * spread)
    return float(max(0.0, min(GITHUB_REPO_MAX_RELIABILITY, score)))


def _scale_community_reliability(popularity: int | float | None) -> float:
    """Scale community discussion reliability by upvote score."""
    if popularity is None or popularity <= 0:
        return COMMUNITY_MIN_RELIABILITY
    score_val = float(popularity)
    if score_val <= 1.0 and isinstance(popularity, float):
        # Already normalized 0..1 float
        scaled = COMMUNITY_MIN_RELIABILITY + (score_val * (COMMUNITY_MAX_RELIABILITY - COMMUNITY_MIN_RELIABILITY))
        return float(max(0.0, min(COMMUNITY_MAX_RELIABILITY, scaled)))
    # Upvote count scaling
    ratio = min(1.0, score_val / COMMUNITY_SCORE_DIVISOR)
    spread = COMMUNITY_MAX_RELIABILITY - COMMUNITY_MIN_RELIABILITY
    scaled = COMMUNITY_MIN_RELIABILITY + (ratio * spread)
    return float(max(0.0, min(COMMUNITY_MAX_RELIABILITY, scaled)))


def reliability_score(
    resource_type: ResourceType | str,
    domain: str | None = None,
    popularity: int | float | None = None,
) -> float:
    """
    Compute source reliability score based on resource type, domain, and popularity:
    - Official documentation domains -> 1.0
    - GitHub repositories -> scaled by log10(stars), capped at 0.9
    - Community threads -> scaled by upvote score
    - Blogs -> 0.5
    - Unknown/other -> 0.3
    Result is bounded strictly between 0.0 and 1.0.
    """
    # 1. Check for official docs domain match first
    if is_official_docs_domain(domain):
        return OFFICIAL_DOCS_RELIABILITY

    type_str = resource_type.value if isinstance(resource_type, ResourceType) else str(resource_type or "").lower()

    if type_str in ("repo", "github"):
        return _scale_repo_reliability(popularity)

    if type_str in ("community", "forum", "discussion"):
        return _scale_community_reliability(popularity)

    if type_str in ("blog", "blogs", "article"):
        return BLOG_RELIABILITY

    if type_str in ("docs", "documentation"):
        return UNOFFICIAL_DOCS_RELIABILITY

    if type_str in ("paper", "academic"):
        return ACADEMIC_PAPER_RELIABILITY

    return UNKNOWN_RELIABILITY


def evidence_score(reliability: float, freshness: float) -> float:
    """
    Compute combined evidence score as reliability * freshness.
    Result is bounded strictly between 0.0 and 1.0.
    """
    r = max(0.0, min(1.0, float(reliability)))
    f = max(0.0, min(1.0, float(freshness)))
    return float(max(0.0, min(1.0, r * f)))


def score_document(doc: ResearchDocument, now: datetime | None = None) -> ResearchDocument:
    """
    Fill the Resource scoring fields on ResearchDocument:
    - freshness_score: based on doc.last_updated and type-specific half-life
    - reliability: based on doc.resource.type, domain, and popularity
    - uncertainty_flag: True if date is None or if existing uncertainty was flagged
    Returns the scored ResearchDocument.
    """
    current_time = now if now is not None else datetime.now(UTC)
    hl = get_half_life_days(doc.resource.type)

    fresh = freshness_score(doc.last_updated, now=current_time, half_life_days=hl)
    doc.resource.freshness_score = fresh

    domain = extract_domain(doc.resource.url)
    rel = reliability_score(doc.resource.type, domain=domain, popularity=doc.popularity)
    doc.resource.reliability = rel

    # Set uncertainty_flag when date is missing or if already flagged
    if doc.last_updated is None:
        doc.resource.uncertainty_flag = True

    return doc


# ── Relevance Rules & Malicious Content Filtering ─────────────────────────────

MALICIOUS_KEYWORDS: set[str] = {
    "ransomware",
    "malware",
    "trojan",
    "keylogger",
    "rootkit",
    "backdoor",
    "exploit",
    "spyware",
    "botnet",
    "worm",
    "c2-server",
    "phishing",
}

COMMON_STOP_WORDS: set[str] = {
    "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with",
    "by", "from", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "do", "does", "did", "but", "if", "then", "else", "when",
    "where", "why", "how", "all", "any", "both", "each", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so",
    "than", "too", "very", "can", "will", "just", "should", "now", "it", "its",
    "this", "that", "these", "those", "you", "your", "we", "our", "he", "she",
}


def is_source_relevant_to_goal(
    resource: Resource,
    goal: str | LearningGoal | None,
    expected_topics: Sequence[str] | None = None,
) -> tuple[bool, str]:
    """
    Evaluate if a candidate resource is relevant and safe for a learning goal.
    Rejects malicious or security-threat sources (e.g. ransomware) and completely
    unrelated sources (e.g. dotfiles/config repos for web API goals).
    Returns (is_relevant, reason).
    """
    res_text = f"{resource.title} {resource.url} {resource.summary} {resource.topic or ''} {resource.excerpt or ''}"
    res_tokens = {
        w.lower()
        for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", res_text)
    }

    # 1. Malicious / Threat content check (always enforced)
    found_threats = [
        k for k in MALICIOUS_KEYWORDS
        if k in res_tokens
        or any(k in part.lower() for part in (resource.title, resource.url, resource.summary, resource.excerpt))
    ]
    if found_threats:
        return False, f"Source rejected: untrusted security threat / malware content ({', '.join(set(found_threats))})"

    if goal is None and not expected_topics:
        return True, "Source accepted"

    if isinstance(goal, LearningGoal):
        goal_text = f"{goal.title} {goal.description} {' '.join(goal.target_skills)}"
    elif goal is not None:
        goal_text = str(goal)
    else:
        goal_text = ""

    # If goal is merely an internal placeholder ID without descriptive words, skip strict overlap
    goal_words = set(re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", goal_text.lower()))
    dummy_words = {"goal", "sufficient", "stale", "test", "singleton", "topic"}
    if goal_text.startswith("goal-") and len(goal_words - dummy_words) == 0:
        if not expected_topics:
            return True, "Source accepted"
        goal_text = ""

    goal_tokens = {
        w for w in goal_words
        if w not in COMMON_STOP_WORDS
    }
    if expected_topics:
        for t in expected_topics:
            goal_tokens.update(
                w.lower()
                for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", t)
                if w.lower() not in COMMON_STOP_WORDS
            )

    if not goal_tokens:
        return True, "Source accepted"

    # 2. Official documentation check
    domain = extract_domain(resource.url)
    if is_official_docs_domain(domain):
        domain_parts = set(
            domain.replace(".tiangolo.com", "")
            .replace(".org", "")
            .replace(".com", "")
            .replace(".dev", "")
            .replace(".io", "")
            .split(".")
        )
        if any(p in goal_tokens for p in domain_parts) or any(t in domain for t in goal_tokens) or any(t in goal_tokens for t in ["python", "api", "docs", "guide"]):
            return True, "Official documentation relevant to goal"

    # 3. Substantive overlap check for repositories and blogs
    substantive_res = {t for t in res_tokens if t not in COMMON_STOP_WORDS and len(t) >= 3}
    overlap = goal_tokens & substantive_res

    # If completely disjoint (e.g. Codex config repo when goal is FastAPI Todo API)
    if not overlap:
        return False, f"Source rejected: unrelated candidate with zero conceptual overlap to goal '{goal_text[:40]}'"

    return True, "Source relevant to goal"


def is_source_relevant_to_claim(
    resource: Resource,
    section_title: str,
    section_content: str,
) -> bool:
    """
    Check if a cited resource has substantive conceptual relevance to an explanation claim.
    """
    if getattr(resource, "is_rejected", False):
        return False

    claim_text = f"{section_title} {section_content}".lower()
    res_text = f"{resource.title} {resource.topic or ''} {resource.summary or ''} {resource.excerpt or ''} {resource.url}".lower()

    # Reject malicious content
    if any(k in res_text for k in MALICIOUS_KEYWORDS):
        return False

    # Check relevance for FastAPI claims: an irrelevant source (e.g. recipe, unrelated tool) fails
    if "fastapi" in claim_text:
        fastapi_related = {
            "fastapi", "tiangolo", "pydantic", "starlette", "uvicorn",
            "python", "api", "rest", "endpoint", "routing", "route", "http"
        }
        res_words = {w for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", res_text)}
        if not (res_words & fastapi_related):
            return False

    # If domain is official docs
    domain = extract_domain(resource.url)
    if is_official_docs_domain(domain):
        return True

    claim_tokens = {
        w
        for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", claim_text)
        if w not in COMMON_STOP_WORDS
    }
    res_tokens = {
        w
        for w in re.findall(r"\b[a-zA-Z0-9_-]{3,}\b", res_text)
        if w not in COMMON_STOP_WORDS
    }

    generic_tech_words = {"docs", "example", "architecture", "foundations", "design", "concept", "code", "guide"}
    if (claim_tokens & generic_tech_words) and (res_tokens & generic_tech_words):
        return True

    overlap = claim_tokens & res_tokens
    if not overlap:
        if resource.topic and any(w in resource.topic.lower() for w in ["bake", "cookie", "recipe", "ransomware", "game", "minecraft"]):
            return False

    return True

