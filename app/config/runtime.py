from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os

from dotenv import load_dotenv


@lru_cache(maxsize=1)
def _load_environment() -> None:
    load_dotenv()


@dataclass(frozen=True)
class Settings:
    provider: str
    api_key: str
    model_name: str
    base_url: str | None
    temperature: float = 0.2


def get_env_int(name: str, default: int, min_value: int | None = None) -> int:
    """Read int env var with default and optional lower bound."""
    _load_environment()
    raw = os.getenv(name, "").strip()
    if not raw:
        value = int(default)
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a valid integer value.") from exc

    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}.")
    return value


def get_env_float(
    name: str,
    default: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Read float env var with default and optional bounds."""
    _load_environment()
    raw = os.getenv(name, "").strip()
    if not raw:
        value = float(default)
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise ValueError(f"{name} must be a valid float value.") from exc

    if min_value is not None and value < min_value:
        raise ValueError(f"{name} must be >= {min_value}.")
    if max_value is not None and value > max_value:
        raise ValueError(f"{name} must be <= {max_value}.")
    return value


def get_env_bool(name: str, default: bool) -> bool:
    """Read bool env var with flexible true/false text values."""
    _load_environment()
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return bool(default)

    if raw in {"1", "true", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "no", "n", "off"}:
        return False

    raise ValueError(f"{name} must be a boolean value (true/false).")


def get_env_str(name: str, default: str = "", lower: bool = False) -> str:
    """Read string env var with default and optional lowercase normalization."""
    _load_environment()
    raw = os.getenv(name, default)
    value = str(raw).strip()
    if not value:
        value = str(default).strip()
    if lower:
        value = value.lower()
    return value


def get_settings() -> Settings:
    """从环境变量读取并校验运行配置。"""
    _load_environment()

    provider = get_env_str("LLM_PROVIDER", "deepseek", lower=True)
    if provider not in {"openai", "deepseek"}:
        raise ValueError("LLM_PROVIDER must be either 'openai' or 'deepseek'.")

    if provider == "deepseek":
        api_key = get_env_str("DEEPSEEK_API_KEY", "")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek.")
        base_url = get_env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        default_model = "deepseek-chat"
    else:
        api_key = get_env_str("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        base_url = get_env_str("OPENAI_BASE_URL", "") or None
        default_model = "gpt-4.1-mini"

    model_name = get_env_str("MODEL_NAME", default_model)
    temperature_raw = get_env_str("TEMPERATURE", "0.2")

    try:
        temperature = float(temperature_raw)
    except ValueError as exc:
        raise ValueError("TEMPERATURE must be a valid float value.") from exc

    return Settings(
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        temperature=temperature,
    )