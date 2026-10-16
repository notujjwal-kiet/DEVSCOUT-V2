"""
tests/test_research_scoring.py

Unit tests for app.research.scoring:
- Halving at exactly one half-life
- None dates handling and uncertainty flag
- Hash stability under whitespace changes
- Hash change on real content change
- Score bounds 0..1 across all scoring functions
- Official docs domain recognition and repo star scaling
"""
from datetime import UTC, datetime, timedelta
import pytest

from app.models.pipeline import ExperienceLevel, Resource, ResourceType
from app.research.models import ResearchDocument
from app.research.scoring import (
    DEFAULT_HALF_LIFE_DAYS,
    DEFAULT_NONE_DATE_FRESHNESS,
    GITHUB_REPO_MAX_RELIABILITY,
    content_changed,
    evidence_score,
    freshness_score,
    hash_content,
    normalize,
    reliability_score,
    score_document,
)


# ── Halving at Exactly One Half-Life ──────────────────────────────────────────

def test_freshness_halving_at_one_half_life():
    """Verify freshness is exactly 0.5 at one half-life, 1.0 at age 0, and 0.25 at two half-lives."""
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    for half_life in [120.0, 180.0, 270.0, 365.0, 900.0]:
        # Age = 0 -> 1.0
        score_zero = freshness_score(last_updated=now, now=now, half_life_days=half_life)
        assert score_zero == pytest.approx(1.0, abs=1e-5)

        # Age = 1 half-life -> exactly 0.5
        updated_1hl = now - timedelta(days=half_life)
        score_1hl = freshness_score(last_updated=updated_1hl, now=now, half_life_days=half_life)
        assert score_1hl == pytest.approx(0.5, abs=1e-5)

        # Age = 2 half-lives -> exactly 0.25
        updated_2hl = now - timedelta(days=2 * half_life)
        score_2hl = freshness_score(last_updated=updated_2hl, now=now, half_life_days=half_life)
        assert score_2hl == pytest.approx(0.25, abs=1e-5)


# ── None Dates & Uncertainty Flag ─────────────────────────────────────────────

def test_freshness_none_date_returns_default():
    """Verify None last_updated returns DEFAULT_NONE_DATE_FRESHNESS (0.3)."""
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    score = freshness_score(last_updated=None, now=now, half_life_days=180.0)
    assert score == pytest.approx(DEFAULT_NONE_DATE_FRESHNESS, abs=1e-5)


def test_score_document_sets_uncertainty_flag_on_none_date():
    """Verify score_document marks uncertainty_flag=True and sets default freshness for None date."""
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    resource = Resource(
        url="https://example.com/blog/intro",
        type=ResourceType.BLOG,
        title="Intro to Testing",
        content_hash="abc123",
        retrieved_at=now,
    )
    doc = ResearchDocument(
        resource=resource,
        text="Sample blog post about testing without timestamp.",
        last_updated=None,
        popularity=None,
    )

    scored = score_document(doc, now=now)
    assert scored.resource.freshness_score == pytest.approx(DEFAULT_NONE_DATE_FRESHNESS, abs=1e-5)
    assert scored.resource.uncertainty_flag is True
    assert scored.resource.reliability == pytest.approx(0.5, abs=1e-5)


def test_score_document_with_valid_date_preserves_uncertainty_flag():
    """Verify score_document calculates freshness and leaves uncertainty_flag False when date is valid."""
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)
    half_life = DEFAULT_HALF_LIFE_DAYS["repo"]  # 270 days
    last_updated = now - timedelta(days=half_life)

    resource = Resource(
        url="https://github.com/tiangolo/fastapi",
        type=ResourceType.REPO,
        title="tiangolo/fastapi",
        content_hash="def456",
        retrieved_at=now,
    )
    doc = ResearchDocument(
        resource=resource,
        text="FastAPI framework code and readme",
        last_updated=last_updated,
        popularity=75000,
    )

    scored = score_document(doc, now=now)
    assert scored.resource.freshness_score == pytest.approx(0.5, abs=1e-5)
    assert scored.resource.uncertainty_flag is False
    assert 0.0 <= scored.resource.reliability <= 0.9


# ── Hash Stability Under Whitespace Changes ────────────────────────────────────

def test_hash_stability_under_whitespace_changes():
    """Verify that tabs, newlines, multiple spaces, and trailing spaces produce identical hashes."""
    text_standard = "def add(x: int, y: int) -> int:\n    return x + y\n"
    text_extra_spaces = "def   add(x:   int, y:   int)   ->   int:    return x + y   "
    text_newlines_tabs = "\n\n\tdef add(x: int,\n\ty: int)\n\t-> int:\n\treturn x + y\n\n"

    h1 = hash_content(text_standard)
    h2 = hash_content(text_extra_spaces)
    h3 = hash_content(text_newlines_tabs)

    assert h1 == h2
    assert h2 == h3
    assert content_changed(h1, text_extra_spaces) is False
    assert content_changed(h1, text_newlines_tabs) is False


# ── Hash Change on Real Content Change ────────────────────────────────────────

def test_hash_change_on_real_content_change():
    """Verify that semantic/textual changes alter the hash and trigger content_changed."""
    text_original = "class Authenticator: def verify(self): return True"
    text_altered = "class Authenticator: def verify(self): return False"

    original_hash = hash_content(text_original)
    altered_hash = hash_content(text_altered)

    assert original_hash != altered_hash
    assert content_changed(original_hash, text_altered) is True
    assert content_changed(altered_hash, text_original) is True


# ── Score Bounds 0..1 ─────────────────────────────────────────────────────────

def test_freshness_score_bounds():
    """Verify freshness_score is strictly bounded between 0.0 and 1.0 for diverse ages."""
    now = datetime(2026, 6, 1, 12, 0, tzinfo=UTC)

    # Future date (negative age) -> clamped to 1.0
    future = now + timedelta(days=100)
    score_future = freshness_score(future, now=now)
    assert 0.0 <= score_future <= 1.0
    assert score_future == pytest.approx(1.0)

    # Distant past date (10,000 days ago) -> very small positive, but >= 0.0
    past = now - timedelta(days=10000)
    score_past = freshness_score(past, now=now)
    assert 0.0 <= score_past <= 1.0

    # Naive vs aware datetime compatibility
    naive_date = datetime(2026, 1, 1, 12, 0)
    score_naive = freshness_score(naive_date, now=now)
    assert 0.0 <= score_naive <= 1.0


def test_reliability_score_bounds_and_categories():
    """Verify reliability_score produces values within [0.0, 1.0] across all categories."""
    # 1. Official docs domains -> exactly 1.0
    assert reliability_score(ResourceType.DOCS, domain="fastapi.tiangolo.com") == 1.0
    assert reliability_score(ResourceType.DOCS, domain="https://docs.python.org/3/") == 1.0
    assert reliability_score(ResourceType.DOCS, domain="developer.mozilla.org") == 1.0

    # 2. GitHub repos -> scaled by log10(stars), capped at 0.9
    assert 0.0 <= reliability_score(ResourceType.REPO, domain="github.com", popularity=0) <= 0.9
    assert 0.0 <= reliability_score(ResourceType.REPO, domain="github.com", popularity=10) <= 0.9
    assert 0.0 <= reliability_score(ResourceType.REPO, domain="github.com", popularity=1000) <= 0.9
    assert reliability_score(ResourceType.REPO, domain="github.com", popularity=100000) == pytest.approx(0.9)
    assert reliability_score(ResourceType.REPO, domain="github.com", popularity=1000000) == pytest.approx(0.9)

    # 3. Community -> by score, bounded in [0.0, 1.0]
    score_comm_zero = reliability_score(ResourceType.COMMUNITY, domain="stackoverflow.com", popularity=0)
    assert 0.0 <= score_comm_zero <= 1.0
    score_comm_high = reliability_score(ResourceType.COMMUNITY, domain="stackoverflow.com", popularity=500)
    assert 0.0 <= score_comm_high <= 1.0
    assert score_comm_high > score_comm_zero

    # 4. Blogs -> 0.5
    assert reliability_score(ResourceType.BLOG, domain="techblog.example.com") == 0.5

    # 5. Unknown -> 0.3
    assert reliability_score("random_type", domain="unknown.xyz") == 0.3


def test_evidence_score_bounds():
    """Verify evidence_score is strictly bounded in [0.0, 1.0] for any input combination."""
    test_cases = [
        (1.0, 1.0, 1.0),
        (0.0, 0.0, 0.0),
        (0.5, 0.5, 0.25),
        (0.9, 0.5, 0.45),
        (1.2, 1.5, 1.0),   # Clamped high inputs
        (-0.5, 0.8, 0.0),  # Clamped negative inputs
    ]

    for rel, fresh, expected in test_cases:
        score = evidence_score(rel, fresh)
        assert 0.0 <= score <= 1.0
        assert score == pytest.approx(expected, abs=1e-5)
