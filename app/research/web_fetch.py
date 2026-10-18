"""
app/research/web_fetch.py

Hardened SSRF-safe web URL validation, redirect resolution, and HTML page fetching.
- Robust IP parsing for standard, decimal, hex, and short (inet_aton) forms.
- DNS resolution via asyncio loop.getaddrinfo checking all A and AAAA records.
- Rejection of private, loopback, link-local, reserved, multicast, unspecified,
  and IPv4-mapped IPv6 addresses.
- Userinfo credential blocking.
- Streamed HTML page fetching with 1 MB body limit, 8-10s timeout, and media type verification.
- Content extraction using beautifulsoup4 + lxml with boilerplate tag decomposition.
"""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
import ipaddress
import socket
import urllib.parse

from bs4 import BeautifulSoup
import httpx
import structlog

from app.models.pipeline import Resource, ResourceType
from app.research.models import ResearchDocument
from app.research.scoring import hash_content, normalize

logger = structlog.get_logger(__name__)

# ── Exceptions & Config ────────────────────────────────────────────────────────

class SSRFError(ValueError):
    """Raised when a URL targets a disallowed scheme, private IP, or loopback address."""


class WebFetchError(Exception):
    """Raised when an HTTP fetch or redirect resolution fails."""


DEFAULT_MAX_REDIRECTS: int = 3
DEFAULT_TIMEOUT_SECONDS: float = 9.0  # 8-10s timeout requirement
MAX_BODY_BYTES: int = 1024 * 1024     # 1 MB max body limit
MAX_TEXT_CHARS: int = 20000

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


# ── SSRF Hardening & IP Parsing ────────────────────────────────────────────────

def parse_ip_host(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """
    Parse a host string if it represents an IP address in any standard or non-standard format:
    - Standard dotted quad (127.0.0.1) or IPv6 (::1, ::ffff:127.0.0.1)
    - Decimal integer forms (e.g. 2130706433 for 127.0.0.1)
    - Hexadecimal integer forms (e.g. 0x7f000001 for 127.0.0.1)
    - Short forms (e.g. 127.1, 0x7f.1) via socket.inet_aton
    Returns an IPv4Address or IPv6Address, or None if the string is a domain name.
    """
    clean_host = host.strip().strip("[]").lower()
    if not clean_host:
        return None

    # 1. Standard ipaddress parser (dotted quad or IPv6)
    try:
        return ipaddress.ip_address(clean_host)
    except ValueError:
        pass

    # 2. Integer or hex integer literals (e.g. 2130706433, 0x7f000001)
    try:
        if clean_host.startswith("0x") or clean_host.isdigit():
            val = int(clean_host, 0)
            if 0 <= val <= 0xFFFFFFFF:
                return ipaddress.IPv4Address(val)
    except (ValueError, OverflowError):
        pass

    # 3. socket.inet_aton for short forms (127.1, 0x7f.1, etc.)
    try:
        raw_bytes = socket.inet_aton(clean_host)
        if len(raw_bytes) == 4:
            return ipaddress.IPv4Address(raw_bytes)
    except (OSError, ValueError):
        pass

    return None


def is_ip_disallowed(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """
    Check if an IP address is disallowed under SSRF constraints:
    Private, loopback, link-local, reserved, multicast, unspecified,
    or an IPv4-mapped IPv6 of any of these.
    """
    # Unpack IPv4-mapped IPv6 addresses (e.g. ::ffff:127.0.0.1)
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped

    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


async def resolve_and_validate(host: str) -> list[ipaddress.IPv4Address | ipaddress.IPv6Address]:
    """
    Resolve and validate a host address against SSRF rules:
    - Parses numeric host forms with ipaddress AND socket.inet_aton-style forms
      (decimal like 2130706433, hex 0x7f000001, short forms like 127.1) and treats them as IPs.
    - Resolves names via asyncio loop.getaddrinfo (all A and AAAA records).
    - Rejects the URL if ANY resolved address is private, loopback, link-local,
      reserved, multicast, unspecified, or an IPv4-mapped IPv6 of those.
    - Blocks userinfo in URLs (user@host).
    Returns list of validated public IP address objects.
    Raises SSRFError if any address is disallowed or resolution fails.

    # NOTE on DNS Rebinding Mitigation:
    # We resolve and validate all A and AAAA records immediately prior to connecting
    # and across every redirect hop.
    # An advanced attacker with a very low TTL DNS server could theoretically return
    # a public IP on resolution and a private IP on subsequent connection.
    # To eliminate this remaining risk completely in environments requiring strict zero-trust,
    # socket connections should bind directly to the validated IP while sending the Host header/SNI.
    # Here, pre-flight validation on every hop enforces rigorous SSRF perimeter defense.
    """
    if not host or not host.strip():
        raise SSRFError("Host is empty")

    clean_host = host.strip().strip("[]").lower()

    if "@" in clean_host:
        raise SSRFError(f"Userinfo credentials detected in host '{host}'")

    if clean_host in BLOCKED_HOSTNAMES or clean_host.endswith(".localhost") or clean_host.endswith(".local"):
        raise SSRFError(f"Host '{host}' is in prohibited hostnames list")

    # Check for direct or non-standard IP literals
    parsed_ip = parse_ip_host(clean_host)
    if parsed_ip is not None:
        if is_ip_disallowed(parsed_ip):
            raise SSRFError(f"Host '{host}' is a prohibited IP address: {parsed_ip}")
        return [parsed_ip]

    # Resolve hostname via asyncio loop.getaddrinfo (all A and AAAA records)
    loop = asyncio.get_running_loop()
    try:
        addr_infos = await loop.getaddrinfo(clean_host, None)
    except Exception as exc:
        raise SSRFError(f"DNS resolution failed for host '{clean_host}': {exc}") from exc

    if not addr_infos:
        raise SSRFError(f"No DNS records resolved for host '{clean_host}'")

    resolved_ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for info in addr_infos:
        sockaddr = info[4]
        ip_str = sockaddr[0]
        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError as exc:
            raise SSRFError(f"Invalid resolved IP '{ip_str}' for host '{clean_host}': {exc}") from exc

        if is_ip_disallowed(ip_obj):
            raise SSRFError(f"Host '{clean_host}' resolved to prohibited IP address: {ip_obj}")
        resolved_ips.append(ip_obj)

    return resolved_ips


def is_safe_url(url: str) -> bool:
    """
    Synchronous SSRF check for schemes, userinfo, IP host forms, and blocked hostnames.
    Does not rely only on simple hostname strings: parses non-standard numeric IP forms
    (decimal, hex, short forms) and verifies IP ranges synchronously.
    """
    if not url:
        return False

    try:
        parsed = urllib.parse.urlparse(url.strip())
    except Exception:
        return False

    if parsed.scheme.lower() not in ("http", "https"):
        return False

    # Block userinfo in URLs (user@host or user:pass@host)
    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        return False

    host = parsed.hostname
    if not host:
        return False

    clean_host = host.strip().strip("[]").lower()
    if clean_host in BLOCKED_HOSTNAMES or clean_host.endswith(".localhost") or clean_host.endswith(".local"):
        return False

    parsed_ip = parse_ip_host(clean_host)
    if parsed_ip is not None:
        return not is_ip_disallowed(parsed_ip)

    # For domains, if synchronous getaddrinfo is available and succeeds, verify records
    try:
        addr_infos = socket.getaddrinfo(clean_host, None)
        for info in addr_infos:
            ip_str = info[4][0]
            ip_obj = ipaddress.ip_address(ip_str)
            if is_ip_disallowed(ip_obj):
                return False
    except Exception:
        pass

    return True


async def validate_url_ssrf_async(url: str) -> str:
    """
    Validate URL safety against SSRF asynchronously:
    - Validates scheme is http or https.
    - Blocks userinfo in URLs.
    - Resolves and validates host (all A/AAAA records).
    Returns the URL if safe, raises SSRFError if unsafe.
    """
    if not url or not url.strip():
        raise SSRFError("URL is empty")

    try:
        parsed = urllib.parse.urlparse(url.strip())
    except Exception as exc:
        raise SSRFError(f"Malformed URL '{url}': {exc}") from exc

    if parsed.scheme.lower() not in ("http", "https"):
        raise SSRFError(f"Prohibited URL scheme '{parsed.scheme}': only http and https allowed")

    if parsed.username is not None or parsed.password is not None or "@" in parsed.netloc:
        raise SSRFError(f"URL contains prohibited userinfo credentials: '{url}'")

    host = parsed.hostname
    if not host:
        raise SSRFError(f"Missing hostname in URL '{url}'")

    await resolve_and_validate(host)
    return url


def validate_url_ssrf(url: str) -> None:
    """Synchronous fallback validation raising SSRFError if unsafe."""
    if not is_safe_url(url):
        raise SSRFError(f"URL violated SSRF safety constraints: '{url}'")


# ── Redirect Resolution ───────────────────────────────────────────────────────

async def resolve_redirects(
    url: str,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    client: httpx.AsyncClient | None = None,
) -> str:
    """
    Resolve potential redirect links step-by-step up to max_redirects.
    Validates initial URL and every redirect hop against resolve_and_validate.
    Uses HEAD first, falling back to GET on 405 Method Not Allowed or transport failure.
    Returns the final resolved URL string.
    Raises SSRFError or WebFetchError on failure.
    """
    await validate_url_ssrf_async(url)

    current_url = url
    headers = {
        "User-Agent": "DEVSCOUT-Learning-Companion/1.0",
        "Accept": "*/*",
    }

    owns_client = client is None
    active_client = client if client is not None else httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)

    try:
        for _ in range(max_redirects):
            await validate_url_ssrf_async(current_url)

            # Try HEAD first
            resp = None
            try:
                resp = await active_client.head(current_url, headers=headers, follow_redirects=False)
                if resp.status_code == 405:
                    resp = await active_client.get(current_url, headers=headers, follow_redirects=False)
            except (httpx.RequestError, Exception):
                try:
                    resp = await active_client.get(current_url, headers=headers, follow_redirects=False)
                except Exception as exc:
                    raise WebFetchError(f"Failed to fetch '{current_url}': {exc}") from exc

            if resp is None:
                raise WebFetchError(f"No response received when resolving '{current_url}'")

            # Check for redirect
            if 300 <= resp.status_code < 400 and "location" in resp.headers:
                location = resp.headers["location"]
                next_url = urllib.parse.urljoin(current_url, location)
                await validate_url_ssrf_async(next_url)
                current_url = next_url
            elif resp.status_code < 400:
                return current_url
            else:
                raise WebFetchError(f"HTTP error {resp.status_code} resolving '{current_url}'")
        else:
            if resp is not None and 300 <= resp.status_code < 400:
                raise WebFetchError(f"Exceeded max redirects ({max_redirects}) for '{url}'")

        await validate_url_ssrf_async(current_url)
        return current_url
    finally:
        if owns_client:
            await active_client.aclose()


# ── HTML Parsing Helpers ──────────────────────────────────────────────────────

def _parse_date_string(date_str: str) -> datetime | None:
    """Attempt to parse date strings into timezone-aware UTC datetime."""
    cleaned = date_str.strip()
    if not cleaned:
        return None

    iso_candidate = cleaned.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(iso_candidate)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        pass

    try:
        d = datetime.strptime(cleaned[:10], "%Y-%m-%d")
        return d.replace(tzinfo=UTC)
    except Exception:
        pass

    return None


def _extract_last_updated(soup: BeautifulSoup) -> datetime | None:
    """Extract article modified/published time from meta or <time> tags."""
    # 1. Check meta tags
    meta_props = [
        {"property": "article:modified_time"},
        {"name": "article:modified_time"},
        {"property": "article:published_time"},
        {"name": "article:published_time"},
        {"property": "og:updated_time"},
    ]
    for attr in meta_props:
        meta = soup.find("meta", attrs=attr)
        if meta and meta.get("content"):
            dt = _parse_date_string(str(meta["content"]))
            if dt is not None:
                return dt

    # 2. Check <time> tag
    time_tag = soup.find("time")
    if time_tag:
        content = time_tag.get("datetime") or time_tag.get_text()
        if content:
            dt = _parse_date_string(str(content))
            if dt is not None:
                return dt

    return None


# ── HTML Page Fetcher ─────────────────────────────────────────────────────────

async def fetch_page(
    url: str,
    resource_type: ResourceType | str = ResourceType.BLOG,
    client: httpx.AsyncClient | None = None,
) -> ResearchDocument:
    """
    Fetch web page through SSRF-safe redirect resolver and extract main content:
    - Resolves redirects safely against SSRF constraints.
    - Enforces 8-10s timeout and 1 MB max body limit (streams and aborts when exceeded).
    - Rejects non-HTML Content-Types (only text/html or application/xhtml+xml allowed).
    - Extracts <title> and main text with beautifulsoup4 + lxml.
    - Drops script, style, nav, footer, aside tags.
    - Collapses whitespace and truncates text to 20,000 characters.
    - Computes content_hash via scoring.normalize + hash_content.
    - Extracts last_updated from meta article:modified_time / article:published_time / <time>.
    - Raises WebFetchError on any error.
    """
    try:
        final_url = await resolve_redirects(url, max_redirects=DEFAULT_MAX_REDIRECTS, client=client)
    except Exception as exc:
        raise WebFetchError(f"Failed to resolve URL '{url}': {exc}") from exc

    owns_client = client is None
    active_client = client if client is not None else httpx.AsyncClient(timeout=DEFAULT_TIMEOUT_SECONDS)

    headers = {
        "User-Agent": "DEVSCOUT-Learning-Companion/1.0",
        "Accept": "text/html,application/xhtml+xml;q=0.9,*/*;q=0.8",
    }

    try:
        # Validate final URL immediately pre-flight before stream
        await validate_url_ssrf_async(final_url)

        try:
            async with active_client.stream("GET", final_url, headers=headers, follow_redirects=False) as response:
                if response.status_code >= 400:
                    raise WebFetchError(f"HTTP error {response.status_code} fetching '{final_url}'")

                content_type = response.headers.get("content-type", "").lower()
                media_type = content_type.split(";")[0].strip()
                if media_type not in ("text/html", "application/xhtml+xml"):
                    raise WebFetchError(
                        f"Disallowed Content-Type '{content_type}' for '{final_url}': "
                        f"only text/html or application/xhtml+xml supported"
                    )

                # Stream response bytes and enforce 1 MB max limit
                chunks: list[bytes] = []
                total_bytes = 0
                async for chunk in response.aiter_bytes():
                    total_bytes += len(chunk)
                    if total_bytes > MAX_BODY_BYTES:
                        raise WebFetchError(
                            f"Response body exceeded 1 MB limit ({total_bytes} bytes) for '{final_url}'"
                        )
                    chunks.append(chunk)

                body_bytes = b"".join(chunks)
        except WebFetchError:
            raise
        except Exception as exc:
            raise WebFetchError(f"HTTP fetch failed for '{final_url}': {exc}") from exc

        body_text = body_bytes.decode("utf-8", errors="replace")

        # Parse with beautifulsoup4 + lxml
        try:
            soup = BeautifulSoup(body_text, "lxml")
        except Exception:
            soup = BeautifulSoup(body_text, "html.parser")

        # Decompose non-content and noise tags: script, style, nav, footer, aside
        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()

        # Extract title
        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()
        if not title and soup.find("h1"):
            title = soup.find("h1").get_text().strip()
        if not title:
            title = final_url

        # Extract main text
        main_elem = soup.find("main") or soup.find("article") or soup.body or soup
        raw_text = main_elem.get_text(separator=" ")
        collapsed_text = normalize(raw_text)
        truncated_text = collapsed_text[:MAX_TEXT_CHARS]

        content_hash_val = hash_content(truncated_text)
        last_updated = _extract_last_updated(soup)
        retrieved_at = datetime.now(UTC)

        res_type = ResourceType(resource_type) if isinstance(resource_type, str) else resource_type

        resource = Resource(
            url=final_url,
            type=res_type,
            title=title,
            content_hash=content_hash_val,
            retrieved_at=retrieved_at,
        )

        return ResearchDocument(
            resource=resource,
            text=truncated_text,
            last_updated=last_updated,
            popularity=None,
        )
    finally:
        if owns_client:
            await active_client.aclose()
