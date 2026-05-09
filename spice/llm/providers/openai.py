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


OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
OPENAI_BASE_URL_ENV = "SPICE_OPENAI_BASE_URL"
OPENAI_ORGANIZATION_ENV = "OPENAI_ORG_ID"
OPENAI_PROJECT_ENV = "OPENAI_PROJECT_ID"
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"


@dataclass(slots=True)
class OpenAILLMProvider(LLMProvider):
    provider_id: str = "openai"
    api_key_env: str = OPENAI_API_KEY_ENV
    base_url_env: str = OPENAI_BASE_URL_ENV
    organization_env: str = OPENAI_ORGANIZATION_ENV
    project_env: str = OPENAI_PROJECT_ENV

    def generate(self, request: LLMRequest, model: LLMModelConfig) -> LLMResponse:
        api_key = _env_value(self.api_key_env)
        if not api_key:
            raise LLMAuthError(f"{self.api_key_env} is required for OpenAI provider.")
        if not model.model_id.strip():
            raise LLMResponseError("OpenAI model_id is required.")

        endpoint = chat_completions_endpoint(
            _env_value(self.base_url_env),
            OPENAI_DEFAULT_BASE_URL,
        )
        timeout_sec = model.timeout_sec if model.timeout_sec is not None else request.timeout_sec
        parsed, latency_ms = post_chat_completions(
            provider_label="OpenAI",
            endpoint=endpoint,
            headers=self._headers(api_key),
            payload=build_chat_payload(request=request, model=model),
            timeout_sec=timeout_sec,
        )
        output_text, finish_reason = extract_choice(parsed, provider_label="OpenAI")
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
            request_id=str(parsed.get("id") or f"oa-{uuid4().hex}"),
        )

    def _headers(self, api_key: str) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        organization = _env_value(self.organization_env)
        if organization:
            headers["OpenAI-Organization"] = organization
        project = _env_value(self.project_env)
        if project:
            headers["OpenAI-Project"] = project
        return headers


def _env_value(name: str) -> str:
    return os.environ.get(name, "").strip()
