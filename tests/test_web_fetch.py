"""
tests/test_web_fetch.py

Unit tests for hardened SSRF safety and fetch_page in app.research.web_fetch:
- Decimal, hex, and short IP forms rejected
- Domain resolving to 127.0.0.1 rejected
- Domain with one public and one private record rejected
- Redirect to a private host rejected
- Userinfo in URL rejected
- Oversized body (> 1 MB) aborted
- Non-HTML content types rejected
- Nav, footer, aside, script, style stripped from HTML
- Normal HTML page parsed with metadata and hash
"""
import asyncio
from datetime import UTC, datetime
import socket
import httpx
import pytest

from app.models.pipeline import ResourceType
from app.research.scoring import hash_content
from app.research.web_fetch import (
    SSRFError,
    WebFetchError,
    fetch_page,
    is_safe_url,
    resolve_and_validate,
    resolve_redirects,
)


def mock_dns(monkeypatch, mapping: dict[str, list[str]]):
    """
    Mock socket.getaddrinfo to return custom IP records for both sync and asyncio lookups.
    mapping maps hostname -> list of IP strings.
    """
    orig_getaddrinfo = socket.getaddrinfo

    def fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        if isinstance(host, str) and host in mapping:
            records = []
            for ip in mapping[host]:
                fam = socket.AF_INET6 if ":" in ip else socket.AF_INET
                records.append((fam, socket.SOCK_STREAM, 6, "", (ip, port or 80)))
            return records
        return orig_getaddrinfo(host, port, family, type, proto, flags)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)


# ── 1. Decimal / Hex / Short IP Forms Rejected ────────────────────────────────

@pytest.mark.asyncio
async def test_numeric_and_short_ip_forms_rejected():
    """Verify non-standard numeric IP representations of loopback/private IPs are detected and rejected."""
    bad_hosts = [
        "2130706433",     # Decimal 127.0.0.1
        "0x7f000001",     # Hex 127.0.0.1
        "127.1",          # Short form 127.0.0.1
        "0x7f.1",         # Mixed hex short form 127.0.0.1
        "10.1",           # Private 10.0.0.1
        "0xa000001",      # Hex 10.0.0.1
        "169.254.169.254",# Link-local metadata
    ]

    for host in bad_hosts:
        assert is_safe_url(f"http://{host}/") is False
        with pytest.raises(SSRFError):
            await resolve_and_validate(host)


# ── 2. Domain Resolving to 127.0.0.1 Rejected ──────────────────────────────────

@pytest.mark.asyncio
async def test_domain_resolving_to_loopback_rejected(monkeypatch):
    """Verify a domain name whose DNS A-record resolves to 127.0.0.1 is rejected."""
    mock_dns(monkeypatch, {"evil-loopback.org": ["127.0.0.1"]})

    with pytest.raises(SSRFError, match="prohibited IP address"):
        await resolve_and_validate("evil-loopback.org")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": "text/html"}, text="<h1>Trap</h1>")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(WebFetchError):
            await fetch_page("http://evil-loopback.org/admin", client=client)


# ── 3. Domain with One Public and One Private Record Rejected ─────────────────

@pytest.mark.asyncio
async def test_domain_with_mixed_public_private_records_rejected(monkeypatch):
    """Verify a domain returning multiple A/AAAA records is rejected if ANY record is private."""
    mock_dns(monkeypatch, {
        "mixed-records.org": [
            "93.184.216.34",  # Public (example.com)
            "10.0.0.1",       # Private RFC1918
        ]
    })

    with pytest.raises(SSRFError, match="prohibited IP address"):
        await resolve_and_validate("mixed-records.org")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": "text/html"}, text="<h1>OK</h1>")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(WebFetchError):
            await fetch_page("http://mixed-records.org/data", client=client)


# ── 4. Redirect to Private Host Rejected ──────────────────────────────────────

@pytest.mark.asyncio
async def test_redirect_to_private_host_rejected(monkeypatch):
    """Verify following a redirect to an internal private address triggers an immediate error."""
    mock_dns(monkeypatch, {
        "safe-entry.org": ["93.184.216.34"],
        "internal-service.local": ["192.168.1.100"],
    })

    def handler(request: httpx.Request) -> httpx.Response:
        if "safe-entry.org" in str(request.url):
            return httpx.Response(302, headers={"Location": "http://192.168.1.100/admin"})
        return httpx.Response(200, headers={"Content-Type": "text/html"}, text="<h1>Admin</h1>")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(WebFetchError):
            await fetch_page("http://safe-entry.org/login", client=client)


# ── 5. Userinfo Rejected ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_userinfo_in_url_rejected():
    """Verify URLs containing user:pass@ or user@ are rejected as SSRF risk."""
    bad_urls = [
        "http://user@public-site.org/path",
        "http://admin:secret@public-site.org/api",
    ]

    for url in bad_urls:
        assert is_safe_url(url) is False
        with pytest.raises(SSRFError):
            await resolve_and_validate("user@public-site.org")


# ── 6. Oversized Body Aborted ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_oversized_body_aborted(monkeypatch):
    """Verify fetch_page streams and aborts if body exceeds 1 MB (1,048,576 bytes)."""
    mock_dns(monkeypatch, {"public-site.org": ["93.184.216.34"]})

    # 1.2 MB response body
    big_content = b"<html><body>" + (b"A" * (1200 * 1024)) + b"</body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"Content-Type": "text/html"},
            content=big_content,
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(WebFetchError, match="exceeded 1 MB limit"):
            await fetch_page("http://public-site.org/huge.html", client=client)


# ── 7. Non-HTML Content-Type Rejected ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_non_html_content_type_rejected(monkeypatch):
    """Verify fetch_page rejects responses that are not text/html or application/xhtml+xml."""
    mock_dns(monkeypatch, {"public-site.org": ["93.184.216.34"]})

    def handler_json(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": "application/json"}, text='{"result": "ok"}')

    transport = httpx.MockTransport(handler_json)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(WebFetchError, match="Disallowed Content-Type"):
            await fetch_page("http://public-site.org/api/data", client=client)


# ── 8. Nav / Footer / Script / Style / Aside Stripped ─────────────────────────

@pytest.mark.asyncio
async def test_nav_footer_aside_script_style_stripped(monkeypatch):
    """Verify non-content elements (nav, footer, aside, script, style) are decomposed."""
    mock_dns(monkeypatch, {"public-site.org": ["93.184.216.34"]})

    html = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>Python Concurrency Guide</title>
        <style>body { background: #000; } .nav-link { color: red; }</style>
      </head>
      <body>
        <nav><a href="/">Home</a> <a href="/docs">Documentation</a></nav>
        <aside>Promoted Courses &amp; Advertisements</aside>
        <main>
          <h1>Understanding Asyncio Event Loops</h1>
          <p>Asyncio provides cooperative multitasking with coroutines and tasks.</p>
        </main>
        <script>console.log("analytics tracking code");</script>
        <footer>&copy; 2026 Tech Corp. All rights reserved.</footer>
      </body>
    </html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": "text/html; charset=utf-8"}, text=html)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        doc = await fetch_page("http://public-site.org/async-guide", resource_type=ResourceType.DOCS, client=client)

    assert doc.resource.title == "Python Concurrency Guide"
    assert "Understanding Asyncio Event Loops" in doc.text
    assert "Asyncio provides cooperative multitasking" in doc.text

    # Verify decomposed elements are stripped
    assert "Home" not in doc.text
    assert "Documentation" not in doc.text
    assert "Promoted Courses" not in doc.text
    assert "analytics tracking" not in doc.text
    assert "All rights reserved" not in doc.text
    assert "background: #000" not in doc.text


# ── 9. Normal HTML Page Parsed ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_normal_page_parsed(monkeypatch):
    """Verify standard HTML page produces a populated ResearchDocument with title, hash, and timestamp."""
    mock_dns(monkeypatch, {"public-site.org": ["93.184.216.34"]})

    html = """
    <!DOCTYPE html>
    <html>
      <head>
        <title>FastAPI Dependency Injection</title>
        <meta property="article:published_time" content="2026-06-15T10:00:00Z" />
      </head>
      <body>
        <article>
          <h1>Dependency Injection in FastAPI</h1>
          <p>FastAPI has a very powerful but intuitive Dependency Injection system.</p>
        </article>
      </body>
    </html>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"Content-Type": "text/html"}, text=html)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        doc = await fetch_page("http://public-site.org/fastapi-di", resource_type=ResourceType.DOCS, client=client)

    assert doc.resource.url == "http://public-site.org/fastapi-di"
    assert doc.resource.title == "FastAPI Dependency Injection"
    assert doc.resource.type == ResourceType.DOCS
    assert "Dependency Injection in FastAPI" in doc.text
    assert doc.last_updated == datetime(2026, 6, 15, 10, 0, tzinfo=UTC)
    assert doc.resource.content_hash == hash_content(doc.text)
    assert doc.resource.retrieved_at is not None


# ── 10. resolve_redirects Multi-hop and Status Tests ──────────────────────────

@pytest.mark.asyncio
async def test_resolve_redirects_three_hops_succeeds():
    """Verify a 3-hop redirect chain succeeds with max_redirects=3."""
    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if url_str.endswith("/hop0"):
            return httpx.Response(302, headers={"Location": "http://example.com/hop1"})
        elif url_str.endswith("/hop1"):
            return httpx.Response(302, headers={"Location": "http://example.com/hop2"})
        elif url_str.endswith("/hop2"):
            return httpx.Response(302, headers={"Location": "http://example.com/hop3"})
        elif url_str.endswith("/hop3"):
            return httpx.Response(200, text="Final Content")
        return httpx.Response(404, text="Unknown")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        final_url = await resolve_redirects("http://example.com/hop0", max_redirects=3, client=client)

    assert final_url == "http://example.com/hop3"


@pytest.mark.asyncio
async def test_resolve_redirects_four_hops_fails_max_redirects_three():
    """Verify a 4-hop redirect chain raises WebFetchError when max_redirects=3."""
    def handler(request: httpx.Request) -> httpx.Response:
        url_str = str(request.url)
        if url_str.endswith("/hop0"):
            return httpx.Response(302, headers={"Location": "http://example.com/hop1"})
        elif url_str.endswith("/hop1"):
            return httpx.Response(302, headers={"Location": "http://example.com/hop2"})
        elif url_str.endswith("/hop2"):
            return httpx.Response(302, headers={"Location": "http://example.com/hop3"})
        elif url_str.endswith("/hop3"):
            return httpx.Response(302, headers={"Location": "http://example.com/hop4"})
        elif url_str.endswith("/hop4"):
            return httpx.Response(200, text="Final Content")
        return httpx.Response(404, text="Unknown")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(WebFetchError, match="Exceeded max redirects"):
            await resolve_redirects("http://example.com/hop0", max_redirects=3, client=client)


@pytest.mark.asyncio
async def test_resolve_redirects_head_403_retries_get_200():
    """Verify that if HEAD returns >= 400 (e.g. 403), it retries with GET and succeeds if GET returns 200."""
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.method)
        if request.method == "HEAD":
            return httpx.Response(403, text="Forbidden HEAD")
        if request.method == "GET":
            return httpx.Response(200, text="OK GET")
        return httpx.Response(405)

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        final_url = await resolve_redirects("http://example.com/page", client=client)

    assert final_url == "http://example.com/page"
    assert calls == ["HEAD", "GET"]


@pytest.mark.asyncio
async def test_resolve_redirects_final_403_401_429_kept():
    """Verify that terminal 401, 403, or 429 returns the last SSRF-validated URL instead of raising."""
    for status in (401, 403, 429):
        def handler(request: httpx.Request, code=status) -> httpx.Response:
            return httpx.Response(code, text="Blocked")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            final_url = await resolve_redirects(f"http://example.com/status-{status}", client=client)
        assert final_url == f"http://example.com/status-{status}"


@pytest.mark.asyncio
async def test_resolve_redirects_final_404_and_410_raise():
    """Verify that terminal 404 and 410 raise WebFetchError."""
    for status in (404, 410):
        def handler(request: httpx.Request, code=status) -> httpx.Response:
            return httpx.Response(code, text="Missing")

        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            with pytest.raises(WebFetchError, match=f"HTTP {status}"):
                await resolve_redirects(f"http://example.com/status-{status}", client=client)
