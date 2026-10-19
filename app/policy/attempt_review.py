"""
app/policy/attempt_review.py

Deterministic, LLM-free learner attempt reviewer.
Inspects submitted code attempts to determine whether they demonstrate genuine effort:
- Checks minimum real code lines (ignoring blanks and comments).
- Checks AST-normalized form (identifiers and literals abstracted) and token Jaccard
  against provided snippets; flags COPIED_SNIPPET above settings threshold.
- Verifies syntax via ast for Python (or parser if available for JS/TS).
- Verifies relevance against current milestone concepts; flags IRRELEVANT if < K references.
"""
from __future__ import annotations

import ast
import hashlib
import re
from typing import Collection

from app.config import get_settings
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


STRUCTURAL_AST_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.If,
)


def has_structural_substance(tree: ast.AST) -> bool:
    """
    Check if Python AST contains structural logic (function, class, loop, or conditional).
    Rejects code that only consists of literal assignments or expressions.
    """
    for node in ast.walk(tree):
        if isinstance(node, STRUCTURAL_AST_NODES):
            return True
    return False


def check_syntax(code: str, language: str = "python") -> tuple[bool, str | None, ast.AST | None]:
    """
    Check code syntax.
    Python checked via ast (returning parsed tree on success).
    JS/TS checked only if an external parser (e.g. esprima) is available; otherwise skipped.
    """
    lang = language.lower().strip()
    if lang in ("python", "py"):
        try:
            tree = ast.parse(code)
            return True, None, tree
        except SyntaxError as exc:
            return False, f"Python SyntaxError at line {exc.lineno}: {exc.msg}", None

    if lang in ("javascript", "js", "typescript", "ts"):
        try:
            import esprima  # type: ignore

            esprima.parseScript(code)
            return True, None, None
        except ImportError:
            # Parser not available in current environment; skip syntax check
            return True, None, None
        except Exception as exc:
            return False, f"JS/TS SyntaxError: {exc}", None

    # Unsupported language for syntax checking: skip check
    return True, None, None


# ── AST Normalization & Similarity ───────────────────────────────────────────

class _AstAbstractor(ast.NodeTransformer):
    """Abstract identifiers and literals to canonical placeholders for similarity comparison."""

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.Name(id="_ID_", ctx=node.ctx)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        self.generic_visit(node)
        node.arg = "_ARG_"
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.name = "_FUNC_"
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        node.name = "_FUNC_"
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        self.generic_visit(node)
        node.name = "_CLASS_"
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        return ast.Constant(value="_LIT_")


def tokenize_code(code: str) -> list[str]:
    """Tokenize code into words, numbers, and operators."""
    return re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*|\d+|[^\w\s]", code.lower())


def abstract_ast(code: str) -> tuple[str | None, list[str]]:
    """
    Parse Python code and return (canonical_unparsed_str, tokens) with identifiers
    and literals abstracted. If syntax error occurs, returns (None, []).
    """
    try:
        tree = ast.parse(code)
        transformer = _AstAbstractor()
        transformed = transformer.visit(tree)
        ast.fix_missing_locations(transformed)
        unparsed = ast.unparse(transformed)
        tokens = tokenize_code(unparsed)
        return unparsed, tokens
    except Exception:
        return None, []


def token_jaccard_similarity(tokens1: Collection[str], tokens2: Collection[str]) -> float:
    """Compute token Jaccard similarity (|A ∩ B| / |A ∪ B|)."""
    set1 = set(tokens1)
    set2 = set(tokens2)
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def calculate_snippet_similarity(code: str, snippet: str, language: str = "python") -> float:
    """
    Compare code attempt against a provided snippet:
    - If normalized exact hash matches -> 1.0
    - If AST-normalized form (identifiers & literals abstracted) matches exactly -> 1.0
    - Token Jaccard of AST-normalized forms
    - Token Jaccard of raw code tokens
    Returns maximum similarity between 0.0 and 1.0.
    """
    if hash_code(code) == hash_code(snippet):
        return 1.0

    raw_tokens_code = tokenize_code(code)
    raw_tokens_snippet = tokenize_code(snippet)
    raw_jaccard = token_jaccard_similarity(raw_tokens_code, raw_tokens_snippet)

    lang = language.lower().strip()
    if lang in ("python", "py"):
        ast_str_code, ast_tokens_code = abstract_ast(code)
        ast_str_snippet, ast_tokens_snippet = abstract_ast(snippet)
        if ast_str_code is not None and ast_str_snippet is not None:
            if ast_str_code == ast_str_snippet:
                return 1.0
            ast_jaccard = token_jaccard_similarity(ast_tokens_code, ast_tokens_snippet)
            return max(raw_jaccard, ast_jaccard)

    return raw_jaccard


def count_concept_references(code: str, milestone_concepts: Collection[str]) -> tuple[int, list[str]]:
    """
    Count how many identifiers/concepts from milestone_concepts are referenced in the attempt code.
    Matches identifier tokens, substring matches, snake_case variants, and collapsed forms.
    """
    code_lower = code.lower()
    code_tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", code_lower))
    matched: list[str] = []

    for concept in milestone_concepts:
        c_clean = concept.strip()
        if not c_clean:
            continue
        c_lower = c_clean.lower()
        c_snake = re.sub(r"[\s\-]+", "_", c_lower)
        c_flat = re.sub(r"[\s\-_]+", "", c_lower)

        if c_lower in code_tokens or c_snake in code_tokens or c_flat in code_tokens:
            matched.append(c_clean)
        elif len(c_lower) >= 3 and c_lower in code_lower:
            matched.append(c_clean)
        else:
            # Multi-word concept check: all substantive words present in tokens
            words = [w for w in re.findall(r"[a-zA-Z0-9_]+", c_lower) if len(w) >= 3]
            if words and all(w in code_tokens for w in words):
                matched.append(c_clean)

    return len(matched), matched


# ── Review Attempt Entrypoint ────────────────────────────────────────────────

def review_attempt(
    code: str,
    language: str = "python",
    previous_snippet_hashes: Collection[str] | None = None,
    previous_snippets: Collection[str] | None = None,
    milestone_concepts: Collection[str] | None = None,
    min_real_lines: int = DEFAULT_MIN_REAL_LINES,
    copy_threshold: float | None = None,
    min_concept_references: int | None = None,
) -> AttemptReview:
    """
    Review learner attempt deterministically.

    Status outcomes:
    - COPIED_SNIPPET: Attempt matches a previously provided snippet hash or exceeds
                      AST-normalized / token Jaccard similarity threshold against snippets.
    - INVALID: Code fails syntax parsing.
    - TRIVIAL: Attempt has fewer than min_real_lines (excluding blanks and comments)
               or lacks structural substance (e.g., only literal assignments).
    - IRRELEVANT: Attempt does not reference at least K concepts/identifiers from the milestone.
    - ACCEPTABLE: Substantive, syntactically valid, relevant original attempt.
    """
    if not code or not code.strip():
        return AttemptReview(
            status=AttemptStatus.TRIVIAL,
            reasons=["Attempt is empty or contains only whitespace."],
        )

    cfg = get_settings()
    thresh = copy_threshold if copy_threshold is not None else cfg.snippet_copy_threshold

    # 1. Plagiarism / Copied snippet detection
    # A. SHA-256 hash match check (backward compatibility)
    attempt_hash = hash_code(code)
    known_hashes = set(previous_snippet_hashes or [])
    if attempt_hash in known_hashes:
        return AttemptReview(
            status=AttemptStatus.COPIED_SNIPPET,
            reasons=["Attempt matches the hash of a previously provided snippet. Implementation must be your own."],
        )

    # B. AST-normalized form and token Jaccard similarity against provided snippets
    if previous_snippets:
        for snip in previous_snippets:
            sim = calculate_snippet_similarity(code, snip, language=language)
            if sim >= thresh:
                return AttemptReview(
                    status=AttemptStatus.COPIED_SNIPPET,
                    reasons=[
                        f"Attempt is too similar to a previously provided snippet ({sim:.1%} similarity >= {thresh:.1%} threshold). "
                        "Implementation must be your own."
                    ],
                )

    # 2. Syntax validation
    is_valid_syntax, syntax_error, parsed_ast = check_syntax(code, language=language)
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

    # 4. Structural substance check for Python
    lang = language.lower().strip()
    if lang in ("python", "py") and parsed_ast is not None:
        if not has_structural_substance(parsed_ast):
            return AttemptReview(
                status=AttemptStatus.TRIVIAL,
                reasons=[
                    "Attempt lacks structural substance: must include at least one function, "
                    "class, loop (for/while), or conditional (if), not just assignments of literals."
                ],
            )

    # 5. Relevance check against milestone concepts
    if milestone_concepts:
        k = min_concept_references if min_concept_references is not None else cfg.min_concept_references_for_relevance
        required_k = min(k, len(milestone_concepts))
        count, matched = count_concept_references(code, milestone_concepts)
        if count < required_k:
            return AttemptReview(
                status=AttemptStatus.IRRELEVANT,
                reasons=[
                    f"Attempt lacks relevance to the current milestone. "
                    f"Referenced {count} milestone concepts ({matched}); at least {required_k} required from {list(milestone_concepts)}."
                ],
            )

    # 6. Acceptable attempt
    return AttemptReview(
        status=AttemptStatus.ACCEPTABLE,
        reasons=[
            f"Attempt contains {real_lines} real code lines with valid syntax, structural logic, and relevance to the milestone."
        ],
    )
