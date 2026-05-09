from __future__ import annotations

import os
from dataclasses import dataclass
from uuid import uuid4

from spice.llm.core.provider import (
    LLMAuthError,
    LLMProvider,
    LLMResponseError,
)
from spice.llm.core.types import LLMModelConfig, LLMRequest, LLMResponse
from spice.llm.providers.chat_completions import (
    build_chat_payload,
    chat_completions_endpoint,
    extract_choice,
    post_chat_completions,
)


OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"
OPENROUTER_BASE_URL_ENV = "SPICE_OPENROUTER_BASE_URL"
OPENROUTER_SITE_URL_ENV = "SPICE_OPENROUTER_SITE_URL"
OPENROUTER_APP_NAME_ENV = "SPICE_OPENROUTER_APP_NAME"
OPENROUTER_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(slots=True)
class OpenRouterLLMProvider(LLMProvider):
    provider_id: str = "openrouter"
    api_key_env: str = OPENROUTER_API_KEY_ENV
    base_url_env: str = OPENROUTER_BASE_URL_ENV
    site_url_env: str = OPENROUTER_SITE_URL_ENV
    app_name_env: str = OPENROUTER_APP_NAME_ENV

    def generate(self, request: LLMRequest, model: LLMModelConfig) -> LLMResponse:
        api_key = _env_value(self.api_key_env)
        if not api_key:
            raise LLMAuthError(f"{self.api_key_env} is required for OpenRouter provider.")
        if not model.model_id.strip():
            raise LLMResponseError("OpenRouter model_id is required.")

        endpoint = chat_completions_endpoint(
            _env_value(self.base_url_env),
            OPENROUTER_DEFAULT_BASE_URL,
        )
        timeout_sec = (
            model.timeout_sec
            if model.timeout_sec is not None
            else request.timeout_sec
        )
        parsed, latency_ms = post_chat_completions(
            provider_label="OpenRouter",
            endpoint=endpoint,
            headers=self._headers(api_key),
            payload=build_chat_payload(request=request, model=model),
            timeout_sec=timeout_sec,
        )
        output_text, finish_reason = extract_choice(parsed, provider_label="OpenRouter")
        usage = parsed.get("usage")
        if not isinstance(usage, dict):
            usage = {}

        return LLMResponse(
            provider_id=self.provider_id,
            model_id=str(parsed.get("model") or model.model_id),
            output_text=output_text,
            raw_payload=parsed,
            finish_reason=finish_reason,
            usage=usage,
            latency_ms=latency_ms,
            request_id=str(parsed.get("id") or f"or-{uuid4().hex}"),
        )

    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        site_url = _env_value(self.site_url_env)
        if site_url:
            headers["HTTP-Referer"] = site_url
        app_name = _env_value(self.app_name_env)
        if app_name:
            headers["X-OpenRouter-Title"] = app_name
        return headers


def _env_value(name: str) -> str:
    return os.environ.get(name, "").strip()
