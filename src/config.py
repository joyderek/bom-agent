from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import tomllib


DEFAULT_MODEL = os.getenv("BOM_AGENT_MODEL", "deepseek-v4-flash")
DEFAULT_BASE_URL = os.getenv("BOM_AGENT_BASE_URL", "https://api.deepseek.com")
DEFAULT_CONFIG_PATH = Path(os.getenv("BOM_AGENT_CONFIG", "config.local.toml"))


def _load_file_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    return data if isinstance(data, dict) else {}


def _resolve_setting(file_config: dict, env_name: str, file_key: str, default: str) -> str:
    if env_name in os.environ:
        return os.environ[env_name]
    value = file_config.get(file_key, default)
    return str(value)


def _resolve_bool(file_config: dict, env_name: str, file_key: str, default: bool) -> bool:
    if env_name in os.environ:
        value = os.environ[env_name]
    else:
        value = file_config.get(file_key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


@dataclass(slots=True)
class AgentConfig:
    api_key: str = ""
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    search_provider: str = "tavily"
    tavily_api_key: str = ""
    serper_api_key: str = ""
    brave_api_key: str = ""
    searxng_base_url: str = ""
    thinking_enabled: bool = False
    temperature: float = 0.0
    model_timeout_seconds: float = 30.0
    model_max_retries: int = 0
    max_pages: int = 8
    max_search_results: int = 6
    max_tool_calls: int = 12
    recursion_limit: int = 24

    @classmethod
    def from_env(cls) -> "AgentConfig":
        file_config = _load_file_config(DEFAULT_CONFIG_PATH)

        api_key = _resolve_setting(file_config, "BOM_AGENT_API_KEY", "api_key", "")
        if not api_key:
            api_key = _resolve_setting(file_config, "DEEPSEEK_API_KEY", "deepseek_api_key", "")
        if not api_key:
            api_key = _resolve_setting(file_config, "OPENAI_API_KEY", "openai_api_key", "")

        tavily_api_key = _resolve_setting(file_config, "TAVILY_API_KEY", "tavily_api_key", "")
        serper_api_key = _resolve_setting(file_config, "SERPER_API_KEY", "serper_api_key", "")
        brave_api_key = _resolve_setting(file_config, "BRAVE_API_KEY", "brave_api_key", "")
        searxng_base_url = _resolve_setting(
            file_config,
            "SEARXNG_BASE_URL",
            "searxng_base_url",
            "",
        )

        return cls(
            api_key=api_key,
            base_url=_resolve_setting(file_config, "BOM_AGENT_BASE_URL", "base_url", DEFAULT_BASE_URL),
            model=_resolve_setting(file_config, "BOM_AGENT_MODEL", "model", DEFAULT_MODEL),
            search_provider=_resolve_setting(
                file_config,
                "BOM_AGENT_SEARCH_PROVIDER",
                "search_provider",
                "tavily",
            ).strip().lower(),
            tavily_api_key=tavily_api_key,
            serper_api_key=serper_api_key,
            brave_api_key=brave_api_key,
            searxng_base_url=searxng_base_url,
            thinking_enabled=_resolve_bool(
                file_config,
                "BOM_AGENT_THINKING_ENABLED",
                "thinking_enabled",
                False,
            ),
            temperature=float(
                _resolve_setting(file_config, "BOM_AGENT_TEMPERATURE", "temperature", "0")
            ),
            model_timeout_seconds=float(
                _resolve_setting(
                    file_config,
                    "BOM_AGENT_MODEL_TIMEOUT_SECONDS",
                    "model_timeout_seconds",
                    "30",
                )
            ),
            model_max_retries=int(
                _resolve_setting(
                    file_config,
                    "BOM_AGENT_MODEL_MAX_RETRIES",
                    "model_max_retries",
                    "0",
                )
            ),
            max_pages=int(_resolve_setting(file_config, "BOM_AGENT_MAX_PAGES", "max_pages", "8")),
            max_search_results=int(
                _resolve_setting(
                    file_config,
                    "BOM_AGENT_MAX_SEARCH_RESULTS",
                    "max_search_results",
                    "6",
                )
            ),
            max_tool_calls=int(
                _resolve_setting(
                    file_config,
                    "BOM_AGENT_MAX_TOOL_CALLS",
                    "max_tool_calls",
                    "12",
                )
            ),
            recursion_limit=int(
                _resolve_setting(
                    file_config,
                    "BOM_AGENT_RECURSION_LIMIT",
                    "recursion_limit",
                    "24",
                )
            ),
        )
