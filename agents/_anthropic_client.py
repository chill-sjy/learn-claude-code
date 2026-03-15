import os
from typing import Any
from anthropic import Anthropic
from langfuse import get_client


def _setup_tracing() -> None:
    """如果配置了 Langfuse 凭证，则启用 OpenTelemetry 自动插桩。"""
    if os.getenv("LANGFUSE_TRACING_ENABLED", "true").lower() in {"0", "false", "no"}:
        return
    if not (os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY")):
        return

    try:
        from opentelemetry.instrumentation.anthropic import AnthropicInstrumentor
        get_client()
        AnthropicInstrumentor().instrument()
    except ImportError:
        pass


# 模块加载时执行一次，全局生效
_setup_tracing()


def build_client() -> Any:
    base_url = os.getenv("ANTHROPIC_BASE_URL")
    api_key = os.getenv("ANTHROPIC_API_KEY")
    auth_token = os.getenv("ANTHROPIC_AUTH_TOKEN")

    if not api_key and not auth_token:
        raise RuntimeError(
            "Missing credentials. Set ANTHROPIC_API_KEY or ANTHROPIC_AUTH_TOKEN in .env"
        )

    kwargs = {}
    if base_url:
        kwargs["base_url"] = base_url
    if api_key:
        kwargs["api_key"] = api_key
    elif auth_token:
        kwargs["auth_token"] = auth_token

    return Anthropic(**kwargs)
