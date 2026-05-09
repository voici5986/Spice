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


DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL_ENV = "SPICE_DEEPSEEK_BASE_URL"
DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"


@dataclass(slots=True)
class DeepSeekLLMProvider(LLMProvider):
    provider_id: str = "deepseek"
    api_key_env: str = DEEPSEEK_API_KEY_ENV
    base_url_env: str = DEEPSEEK_BASE_URL_ENV

    def generate(self, request: LLMRequest, model: LLMModelConfig) -> LLMResponse:
        api_key = _env_value(self.api_key_env)
        if not api_key:
            raise LLMAuthError(f"{self.api_key_env} is required for DeepSeek provider.")
        if not model.model_id.strip():
            raise LLMResponseError("DeepSeek model_id is required.")

        endpoint = chat_completions_endpoint(
            _env_value(self.base_url_env),
            DEEPSEEK_DEFAULT_BASE_URL,
        )
        timeout_sec = model.timeout_sec if model.timeout_sec is not None else request.timeout_sec
        parsed, latency_ms = post_chat_completions(
            provider_label="DeepSeek",
            endpoint=endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            payload=build_chat_payload(request=request, model=model),
            timeout_sec=timeout_sec,
        )
        output_text, finish_reason = extract_choice(parsed, provider_label="DeepSeek")
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
            request_id=str(parsed.get("id") or f"ds-{uuid4().hex}"),
        )


def _env_value(name: str) -> str:
    return os.environ.get(name, "").strip()
