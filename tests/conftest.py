import asyncio
import asyncio.base_events
import ipaddress
import socket
import pytest
from app.api.auth import get_rate_limiter
from app.config import get_settings
from app.research.cache import reset_research_cache

from app.llm.rate_gate import set_rate_gate
from app.llm.registry import set_model_registry

DEFAULT_MOCK_PUBLIC_IP = "93.184.216.34"


class FakeClock:
    """Controllable monotonic clock and asyncio.sleep simulator for speed and precision."""

    def __init__(self, start_time: float = 1000.0):
        self.current_time = start_time
        self.sleep_calls: list[float] = []

    def monotonic(self) -> float:
        return self.current_time

    async def sleep(self, seconds: float) -> None:
        self.sleep_calls.append(seconds)
        self.current_time += seconds

    def advance(self, seconds: float) -> None:
        self.current_time += seconds


@pytest.fixture
def fake_clock(monkeypatch):
    """Fixture providing a controllable FakeClock patching time.monotonic and asyncio.sleep."""
    clock = FakeClock()
    monkeypatch.setattr("time.monotonic", clock.monotonic)
    monkeypatch.setattr("asyncio.sleep", clock.sleep)
    return clock


@pytest.fixture(autouse=True)
def test_isolation(tmp_path, monkeypatch):
    """
    Autouse test isolation fixture:
    (a) Point gemini_quota_file, request_cache_file, and research cache file/dir
        at tmp_path for every test.
    (b) Clear settings and research caches before and after each test.
    (c) Reset rate limiter, rate gate, and model registry singletons.
    (d) Set dummy GEMINI_API_KEY and GEMINI_MODEL via monkeypatch so no test needs .env.
    """
    quota_file = str(tmp_path / "quota.json")
    request_cache_file = str(tmp_path / "responses.json")
    research_cache_file = str(tmp_path / "research_cache.json")
    research_cache_dir = str(tmp_path / "research_cache")

    monkeypatch.setenv("GEMINI_QUOTA_FILE", quota_file)
    monkeypatch.setenv("DEVSCOUT_GEMINI_QUOTA_FILE", quota_file)
    monkeypatch.setenv("REQUEST_CACHE_FILE", request_cache_file)
    monkeypatch.setenv("DEVSCOUT_REQUEST_CACHE_FILE", request_cache_file)
    monkeypatch.setenv("RESEARCH_CACHE_FILE", research_cache_file)
    monkeypatch.setenv("DEVSCOUT_RESEARCH_CACHE_FILE", research_cache_file)
    monkeypatch.setenv("RESEARCH_CACHE_DIR", research_cache_dir)
    monkeypatch.setenv("DEVSCOUT_RESEARCH_CACHE_DIR", research_cache_dir)

    monkeypatch.setenv("GEMINI_API_KEY", "dummy-gemini-test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")

    get_rate_limiter().clear()
    set_rate_gate(None)
    set_model_registry(None)
    get_settings.cache_clear()
    reset_research_cache()

    yield

    get_rate_limiter().clear()
    set_rate_gate(None)
    set_model_registry(None)
    get_settings.cache_clear()
    reset_research_cache()


@pytest.fixture(autouse=True)
def mock_dns_resolver(monkeypatch):
    """
    Autouse fixture that monkeypatches the DNS resolver (socket.getaddrinfo
    and asyncio loop.getaddrinfo) to return a fixed public IP (93.184.216.34)
    for normal test hostnames, preventing tests from touching the real network.
    Tests that test private or loopback resolution keep their own explicit mocks.
    """
    def fake_socket_getaddrinfo(host, port=0, family=0, type=0, proto=0, flags=0):
        effective_port = port if (port is not None and port != 0) else 80
        clean_host = str(host).strip().strip("[]").lower() if host is not None else ""

        if not clean_host:
            raise socket.gaierror(socket.EAI_NONAME, "Name or service not known")

        # Localhost / loopback standard resolution
        if clean_host == "localhost" or clean_host.endswith(".localhost"):
            return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("127.0.0.1", effective_port))]

        # If already an IP address, return it directly
        try:
            ip_obj = ipaddress.ip_address(clean_host)
            fam = socket.AF_INET6 if ip_obj.version == 6 else socket.AF_INET
            return [(fam, socket.SOCK_STREAM, 6, "", (str(ip_obj), effective_port))]
        except ValueError:
            pass

        # For normal test hostnames, return fixed public IP
        fam = socket.AF_INET6 if family == socket.AF_INET6 else socket.AF_INET
        return [(fam, socket.SOCK_STREAM, 6, "", (DEFAULT_MOCK_PUBLIC_IP, effective_port))]

    async def fake_loop_getaddrinfo(self, host, port=0, *, family=0, type=0, proto=0, flags=0):
        # Call socket.getaddrinfo directly so test-level monkeypatches on socket.getaddrinfo are respected
        return socket.getaddrinfo(host, port, family, type, proto, flags)

    monkeypatch.setattr(socket, "getaddrinfo", fake_socket_getaddrinfo)
    monkeypatch.setattr(asyncio.base_events.BaseEventLoop, "getaddrinfo", fake_loop_getaddrinfo)
    monkeypatch.setattr(asyncio.AbstractEventLoop, "getaddrinfo", fake_loop_getaddrinfo)

    orig_connect = socket.socket.connect

    def fake_socket_connect(self, address):
        if isinstance(address, tuple) and len(address) >= 2:
            ip, port = address[0], address[1]
            if str(ip) == DEFAULT_MOCK_PUBLIC_IP:
                raise ConnectionRefusedError(
                    f"Connection to mock public IP {DEFAULT_MOCK_PUBLIC_IP}:{port} refused (fast-fail mock)."
                )
        return orig_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", fake_socket_connect)

    orig_create_connection = asyncio.base_events.BaseEventLoop.create_connection

    async def fake_create_connection(self, protocol_factory, host=None, port=None, *args, **kwargs):
        clean_host = str(host).strip().strip("[]").lower() if host is not None else ""
        if clean_host == DEFAULT_MOCK_PUBLIC_IP:
            raise ConnectionRefusedError(
                f"Connection to mock public IP {DEFAULT_MOCK_PUBLIC_IP}:{port} refused (fast-fail mock)."
            )
        return await orig_create_connection(self, protocol_factory, host=host, port=port, *args, **kwargs)

    monkeypatch.setattr(asyncio.base_events.BaseEventLoop, "create_connection", fake_create_connection)
    monkeypatch.setattr(asyncio.AbstractEventLoop, "create_connection", fake_create_connection)


# Fast SSL context cache for tests (prevents Windows CryptoAPI certificate store rescans on every client/SSL init)
try:
    import ssl
    import certifi

    _orig_create_default_context = ssl.create_default_context
    _default_ssl_contexts: dict = {}

    def _fast_create_default_context(purpose=ssl.Purpose.SERVER_AUTH, *, cafile=None, capath=None, cadata=None):
        key = (purpose, cafile, capath, cadata)
        if key not in _default_ssl_contexts:
            _default_ssl_contexts[key] = _orig_create_default_context(
                purpose, cafile=cafile, capath=capath, cadata=cadata
            )
        return _default_ssl_contexts[key]

    ssl.create_default_context = _fast_create_default_context

    from google.genai._api_client import BaseApiClient

    _cached_test_ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    BaseApiClient._ensure_httpx_ssl_ctx = staticmethod(
        lambda options, vertexai=False: ({"verify": _cached_test_ssl_ctx}, {"verify": _cached_test_ssl_ctx})
    )
    BaseApiClient._ensure_aiohttp_ssl_ctx = staticmethod(
        lambda options, vertexai=False: {"ssl": _cached_test_ssl_ctx}
    )
    BaseApiClient._ensure_websocket_ssl_ctx = staticmethod(
        lambda options, vertexai=False: {"ssl": _cached_test_ssl_ctx}
    )
except Exception:
    pass
