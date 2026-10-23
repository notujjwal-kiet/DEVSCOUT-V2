"""
app/providers/registry.py

Configures multi-provider registry from backend environment variables and Settings.
No hardcoded secrets. Never logs API keys.
Dynamically tracks configuration changes and provides safe diagnostic reporting
(distinguishing configuration, initialization, authentication, model access, and quota).
"""
from __future__ import annotations

import os
from typing import Any
import structlog

from app.providers.base import BaseChatProvider
from app.providers.gemini_provider import GeminiChatProvider
from app.providers.openai_compatible import OpenAICompatibleProvider

logger = structlog.get_logger(__name__)


class ProviderRegistry:
    """
    Registry managing all configured AI providers:
    - Groq (Primary chat)
    - Cerebras (Configurable fallback)
    - Google Gemini (Research provider)
    - Ollama (Optional local fallback)
    """

    def __init__(self) -> None:
        self._providers: dict[str, BaseChatProvider] = {}
        self.reload()

    def reload(self) -> None:
        """Initialize or reload providers from current environment and config."""
        try:
            from dotenv import load_dotenv
            load_dotenv(override=True)
            from app.config import get_settings
            get_settings.cache_clear()
        except Exception:
            pass

        from app.config import get_settings
        cfg = get_settings()
        self._providers.clear()

        # 1. Groq (Primary Chat)
        groq_key = getattr(cfg, "groq_api_key", None) or os.getenv("GROQ_API_KEY")
        groq_base_url = (
            getattr(cfg, "groq_base_url", None)
            or os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
        )
        groq_model = (
            getattr(cfg, "groq_model", None)
            or os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
        )
        groq_timeout = getattr(cfg, "groq_timeout_seconds", 30.0)

        self._providers["groq"] = OpenAICompatibleProvider(
            name="groq",
            base_url=groq_base_url,
            model=groq_model,
            api_key=groq_key,
            timeout_seconds=groq_timeout,
            tier="primary",
            is_local=False,
            price_in_per_1m=0.05,
            price_out_per_1m=0.08,
        )

        # 2. Cerebras (Configurable Fallback)
        cerebras_key = getattr(cfg, "cerebras_api_key", None) or os.getenv("CEREBRAS_API_KEY")
        cerebras_base_url = (
            getattr(cfg, "cerebras_base_url", None)
            or os.getenv("CEREBRAS_BASE_URL", "https://api.cerebras.ai/v1")
        )
        cerebras_model = (
            getattr(cfg, "cerebras_model", None)
            or os.getenv("CEREBRAS_MODEL", "llama3.1-8b")
        )
        cerebras_timeout = getattr(cfg, "cerebras_timeout_seconds", 30.0)

        self._providers["cerebras"] = OpenAICompatibleProvider(
            name="cerebras",
            base_url=cerebras_base_url,
            model=cerebras_model,
            api_key=cerebras_key,
            timeout_seconds=cerebras_timeout,
            tier="fallback",
            is_local=False,
            price_in_per_1m=0.10,
            price_out_per_1m=0.10,
        )

        # 3. Google Gemini (Research Provider)
        gemini_key = getattr(cfg, "gemini_api_key", None) or os.getenv("GEMINI_API_KEY")
        gemini_model = getattr(cfg, "gemini_model", "gemini-2.5-flash")
        gemini_timeout = getattr(cfg, "gemini_timeout_seconds", 30.0)

        self._providers["gemini"] = GeminiChatProvider(
            name="gemini",
            model=gemini_model,
            api_key=gemini_key,
            timeout_seconds=gemini_timeout,
            tier="research",
            rpd_limit=getattr(cfg, "gemini_rpd_limit", 20),
            rpm_limit=getattr(cfg, "gemini_rpm", 5),
            price_in_per_1m=0.075,
            price_out_per_1m=0.30,
        )

        # 4. Ollama (Optional Local Fallback)
        ollama_enabled_val = getattr(cfg, "ollama_enabled", False)
        if isinstance(ollama_enabled_val, str):
            ollama_enabled = ollama_enabled_val.lower() in ("true", "1")
        else:
            ollama_enabled = bool(ollama_enabled_val or os.getenv("OLLAMA_ENABLED", "false").lower() in ("true", "1"))

        ollama_base_url = (
            getattr(cfg, "ollama_base_url", None)
            or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        )
        ollama_model = (
            getattr(cfg, "ollama_model", None)
            or os.getenv("OLLAMA_MODEL", "llama3.2")
        )
        ollama_timeout = getattr(cfg, "ollama_timeout_seconds", 60.0)

        if ollama_enabled:
            self._providers["ollama"] = OpenAICompatibleProvider(
                name="ollama",
                base_url=ollama_base_url,
                model=ollama_model,
                api_key=None,
                timeout_seconds=ollama_timeout,
                tier="local",
                is_local=True,
                price_in_per_1m=0.0,
                price_out_per_1m=0.0,
            )

        logger.info(
            "providers_registry_loaded",
            providers=list(self._providers.keys()),
            groq_available=self._providers["groq"].is_available(),
            cerebras_available=self._providers["cerebras"].is_available(),
            gemini_available=self._providers["gemini"].is_available(),
            ollama_enabled=ollama_enabled,
        )

    def get_provider(self, name: str) -> BaseChatProvider | None:
        """Lookup provider by lowercase name, reloading if unconfigured key is now present in env."""
        key = name.lower().strip()
        prov = self._providers.get(key)
        if prov is None or not prov.is_available():
            # Check if environment was updated with key
            env_var = f"{key.upper()}_API_KEY"
            if os.getenv(env_var):
                self.reload()
                prov = self._providers.get(key)
        return prov

    def list_providers(self) -> list[BaseChatProvider]:
        """Return all registered providers in priority order."""
        return list(self._providers.values())

    def get_all_models_info(self) -> list[dict[str, Any]]:
        """Return public metadata objects for all registered providers (for GET /api/models)."""
        self.reload()
        return [p.get_info() for p in self._providers.values()]

    async def diagnose_provider(self, name: str) -> dict[str, Any]:
        """
        Diagnose provider status without exposing credentials.
        Returns failure category: 'configuration', 'initialization', 'authentication',
        'model_access', 'quota', or 'ready'.
        """
        self.reload()
        prov = self.get_provider(name)
        if not prov:
            return {
                "provider": name,
                "available": False,
                "category": "initialization",
                "message": f"Provider '{name}' not found in registry",
            }
        if isinstance(prov, OpenAICompatibleProvider):
            ok, category = await prov.verify_availability()
            return {
                "provider": name,
                "model": prov.model,
                "available": ok,
                "category": category,
                "status": prov.diagnostic_status,
                "detail": prov.diagnostic_detail,
            }
        return {
            "provider": name,
            "model": prov.model,
            "available": prov.is_available(),
            "category": "ready" if prov.is_available() else "configuration",
            "message": "Configured and available" if prov.is_available() else "Missing API credentials",
        }


_registry_instance: ProviderRegistry | None = None


def get_provider_registry() -> ProviderRegistry:
    """Return singleton ProviderRegistry instance."""
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ProviderRegistry()
    return _registry_instance
