from dataclasses import dataclass
import os

from dotenv import load_dotenv


# 统一封装运行时配置，便于在链路层一次性读取。
@dataclass(frozen=True)
class Settings:
    provider: str
    api_key: str
    model_name: str
    base_url: str | None
    temperature: float = 0.2


def get_settings() -> Settings:
    """从环境变量读取并校验运行配置。"""
    # 先加载 .env 文件，再读取环境变量。
    load_dotenv()

    # 选择模型提供方，当前支持 openai / deepseek。
    provider = os.getenv("LLM_PROVIDER", "deepseek").strip().lower()
    if provider not in {"openai", "deepseek"}:
        raise ValueError("LLM_PROVIDER must be either 'openai' or 'deepseek'.")

    # 按提供方读取不同的 key、base_url 和默认模型名。
    if provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is required when LLM_PROVIDER=deepseek.")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip()
        default_model = "deepseek-chat"
    else:
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
        base_url = os.getenv("OPENAI_BASE_URL", "").strip() or None
        default_model = "gpt-4.1-mini"

    # 如果用户未显式指定模型，则使用对应提供方的默认模型。
    model_name = os.getenv("MODEL_NAME", default_model).strip()
    temperature_raw = os.getenv("TEMPERATURE", "0.2").strip()

    # 将温度参数转换为浮点数，输入非法时给出明确错误。
    try:
        temperature = float(temperature_raw)
    except ValueError as exc:
        raise ValueError("TEMPERATURE must be a valid float value.") from exc

    # 返回不可变配置对象，避免运行中被意外修改。
    return Settings(
        provider=provider,
        api_key=api_key,
        model_name=model_name,
        base_url=base_url,
        temperature=temperature,
    )
