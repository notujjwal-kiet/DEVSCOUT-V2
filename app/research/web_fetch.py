"""
app/research/web_fetch.py

SSRF-safe web URL validation and redirect resolution logic.
Enforces safe schemes, rejects loopback, private, link-local, and cloud metadata addresses,
and resolves redirect chains with hop-by-hop SSRF validation up to max_redirects.
"""
from __future__ import annotations

import ipaddress
import urllib.parse

import httpx
import structlog

logger = structlog.get_logger(__name__)

# ── SSRF Exceptions & Constants ───────────────────────────────────────────────

class SSRFError(ValueError):
    """Raised when a URL targets a disallowed scheme, private IP, or loopback address."""


class WebFetchError(Exception):
    """Raised when an HTTP fetch or redirect resolution fails."""


DEFAULT_MAX_REDIRECTS: int = 3
DEFAULT_TIMEOUT_SECONDS: float = 8.0

BLOCKED_HOSTNAMES: set[str] = {
    "localhost",
    "127.0.0.1",
    "::1",
    "0.0.0.0",
    "metadata.google.internal",
    "metadata.platform.internal",
    "instance-data",
    "169.254.169.254",
}


# ── SSRF Validation ───────────────────────────────────────────────────────────

def is_safe_url(url: str) -> bool:
    """
    Validate that a URL does not violate SSRF constraints:
    - Scheme must be http or https.
    - Hostname must not be in blocked hostnames or end with .localhost/.local.
    - If host is an IP address, must not be private, loopback, link-local, or reserved.
    """
    if not url:
        return False

    try:
        parsed = urllib.parse.urlparse(url.strip())
    except Exception:
        return False

    if parsed.scheme.lower() not in ("http", "https"):
        return False

    host = parsed.hostname
    if not host:
        return False

    host_lower = host.lower().strip("[]")

    if host_lower in BLOCKED_HOSTNAMES:
        return False
    if host_lower.endswith(".localhost") or host_lower.endswith(".local"):
        return False

    # Check for raw IP addresses
    try:
        ip = ipaddress.ip_address(host_lower)
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_multicast
            or ip.is_unspecified
        ):
            return False
    except ValueError:
        # Host is a domain name, not an IP literal
        pass

    return True


def validate_url_ssrf(url: str) -> None:
    """Validate URL safety against SSRF, raising SSRFError if unsafe."""
    if not is_safe_url(url):
        raise SSRFError(f"URL violated SSRF safety constraints: '{url}'")


# ── Redirect Resolution ───────────────────────────────────────────────────────

async def resolve_redirects(
    url: str,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    client: httpx.AsyncClient | None = None,
) -> str:
    """
    Resolve potential redirect links (e.g. search grounding redirect URLs) step-by-step
    up to max_redirects. Each hop is strictly validated against SSRF rules before request.
    Uses HEAD first, falling back to GET if HEAD returns 405 Method Not Allowed or fails.
    Returns the final resolved URL string.
    Raises SSRFError or WebFetchError on failure.
    """
    validate_url_ssrf(url)

    current_url = url
    headers = {
        "User-Agent": "DEVSCOUT-Learning-Companion/1.0",
        "Accept": "*/*",
    }

    owns_client = client is None
    active_client = client if client is not None else httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)

    try:
        for _ in range(max_redirects):
            validate_url_ssrf(current_url)

            # Try HEAD first
            resp = None
            try:
                resp = await active_client.head(current_url, headers=headers, follow_redirects=False)
                if resp.status_code == 405:  # Method Not Allowed
                    resp = await active_client.get(current_url, headers=headers, follow_redirects=False)
            except (httpx.RequestError, Exception):
                # Fallback to GET on any transport issue with HEAD
                try:
                    resp = await active_client.get(current_url, headers=headers, follow_redirects=False)
                except Exception as exc:
                    raise WebFetchError(f"Failed to fetch '{current_url}': {exc}") from exc

            if resp is None:
                raise WebFetchError(f"No response received when resolving '{current_url}'")

            # Check if this response is a redirect
            if 300 <= resp.status_code < 400 and "location" in resp.headers:
                location = resp.headers["location"]
                next_url = urllib.parse.urljoin(current_url, location)
                validate_url_ssrf(next_url)
                current_url = next_url
            elif resp.status_code < 400:
                # Success - reached terminal target URL
                return current_url
            else:
                # 4xx or 5xx error
                raise WebFetchError(f"HTTP error {resp.status_code} resolving '{current_url}'")
        else:
            # Reached max redirects without terminal response
            if resp is not None and 300 <= resp.status_code < 400:
                raise WebFetchError(f"Exceeded max redirects ({max_redirects}) for '{url}'")

        validate_url_ssrf(current_url)
        return current_url
    finally:
        if owns_client:
            await active_client.aclose()
