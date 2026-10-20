import asyncio
import asyncio.base_events
import ipaddress
import socket
import pytest
from app.api.auth import get_rate_limiter
from app.config import get_settings
from app.research.cache import reset_research_cache

DEFAULT_MOCK_PUBLIC_IP = "93.184.216.34"


@pytest.fixture(autouse=True)
def test_isolation(monkeypatch):
    """
    Autouse test isolation fixture:
    (a) reset the rate limiter before each test
    (b) set GEMINI_API_KEY and GEMINI_MODEL via monkeypatch to dummy values so no test needs .env
    (c) clear settings and research caches before and after each test
    """
    get_rate_limiter().clear()
    monkeypatch.setenv("GEMINI_API_KEY", "dummy-gemini-test-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    get_settings.cache_clear()
    reset_research_cache()

    yield

    get_rate_limiter().clear()
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
