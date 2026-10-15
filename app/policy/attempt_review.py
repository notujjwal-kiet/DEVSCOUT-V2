"""
app/policy/attempt_review.py

Deterministic, LLM-free learner attempt reviewer.
Inspects submitted code attempts to determine whether they demonstrate genuine effort:
- Checks minimum real code lines (ignoring blanks and comments).
- Checks SHA-256 normalized hash against previously given snippet hashes to catch copied code.
- Verifies syntax via ast for Python (or parser if available for JS/TS).
"""
from __future__ import annotations

import ast
import hashlib
from typing import Collection

from app.models.pipeline import AttemptReview, AttemptStatus

DEFAULT_MIN_REAL_LINES = 10


def normalize_code(code: str) -> str:
    """Normalize code for content hashing by stripping whitespace and empty lines."""
    lines = [line.strip() for line in code.splitlines() if line.strip()]
    return "\n".join(lines)


def hash_code(code: str) -> str:
    """Calculate SHA-256 digest of normalized code."""
    return hashlib.sha256(normalize_code(code).encode("utf-8")).hexdigest()


def count_real_lines(code: str, language: str = "python") -> int:
    """
    Count real code lines, ignoring blanks, comment-only lines, and standalone docstrings.
    """
    lang = language.lower().strip()
    real_lines = 0

    for raw_line in code.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        # Python comment check
        if lang in ("python", "py"):
            if line.startswith("#"):
                continue
            if line.startswith('"""') and line.endswith('"""') and len(line) >= 6:
                continue
            if line.startswith("'''") and line.endswith("'''") and len(line) >= 6:
                continue
        # JS/TS comment check
        elif lang in ("javascript", "js", "typescript", "ts"):
            if line.startswith("//") or line.startswith("/*") or line.startswith("*"):
                continue
        else:
            # General fallback comments
            if line.startswith("#") or line.startswith("//"):
                continue

        real_lines += 1

    return real_lines


def check_syntax(code: str, language: str = "python") -> tuple[bool, str | None]:
    """
    Check code syntax.
    Python checked via ast.
    JS/TS checked only if an external parser (e.g. esprima) is available; otherwise skipped.
    """
    lang = language.lower().strip()
    if lang in ("python", "py"):
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as exc:
            return False, f"Python SyntaxError at line {exc.lineno}: {exc.msg}"

    if lang in ("javascript", "js", "typescript", "ts"):
        try:
            import esprima  # type: ignore

            esprima.parseScript(code)
            return True, None
        except ImportError:
            # Parser not available in current environment; skip syntax check
            return True, None
        except Exception as exc:
            return False, f"JS/TS SyntaxError: {exc}"

    # Unsupported language for syntax checking: skip check
    return True, None


def review_attempt(
    code: str,
    language: str = "python",
    previous_snippet_hashes: Collection[str] | None = None,
    min_real_lines: int = DEFAULT_MIN_REAL_LINES,
) -> AttemptReview:
    """
    Review learner attempt deterministically.

    Status outcomes:
    - COPIED_SNIPPET: Attempt matches a previously provided snippet hash.
    - INVALID: Code fails syntax parsing.
    - TRIVIAL: Attempt has fewer than min_real_lines (excluding blanks and comments).
    - ACCEPTABLE: Substantive, syntactically valid original attempt.
    """
    if not code or not code.strip():
        return AttemptReview(
            status=AttemptStatus.TRIVIAL,
            reasons=["Attempt is empty or contains only whitespace."],
        )

    # 1. Plagiarism / Copied snippet detection via SHA-256
    attempt_hash = hash_code(code)
    known_hashes = set(previous_snippet_hashes or [])
    if attempt_hash in known_hashes:
        return AttemptReview(
            status=AttemptStatus.COPIED_SNIPPET,
            reasons=["Attempt matches the hash of a previously provided snippet. Implementation must be your own."],
        )

    # 2. Syntax validation
    is_valid_syntax, syntax_error = check_syntax(code, language=language)
    if not is_valid_syntax:
        return AttemptReview(
            status=AttemptStatus.INVALID,
            reasons=[syntax_error or "Code attempt contains invalid syntax."],
        )

    # 3. Non-triviality check (minimum real code lines)
    real_lines = count_real_lines(code, language=language)
    if real_lines < min_real_lines:
        return AttemptReview(
            status=AttemptStatus.TRIVIAL,
            reasons=[
                f"Attempt contains only {real_lines} real code lines; minimum required is {min_real_lines}. "
                "Blanks and comments are excluded."
            ],
        )

    # 4. Acceptable attempt
    return AttemptReview(
        status=AttemptStatus.ACCEPTABLE,
        reasons=[
            f"Attempt contains {real_lines} real code lines with valid syntax and no snippet copying detected."
        ],
    )
