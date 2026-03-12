import os

from anthropic import Anthropic


def build_client() -> Anthropic:
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
