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


XIAOMI_API_KEY_ENV = "XIAOMI_API_KEY"
XIAOMI_BASE_URL_ENV = "XIAOMI_BASE_URL"
MIMO_API_KEY_ENV = "MIMO_API_KEY"
MIMO_BASE_URL_ENV = "SPICE_MIMO_BASE_URL"
MIMO_API_KEY_ENV_ALIASES = (XIAOMI_API_KEY_ENV, MIMO_API_KEY_ENV)
MIMO_BASE_URL_ENV_ALIASES = (XIAOMI_BASE_URL_ENV, MIMO_BASE_URL_ENV)
MIMO_DEFAULT_BASE_URL = "https://token-plan-cn.xiaomimimo.com/v1"


@dataclass(slots=True)
class MiMoLLMProvider(LLMProvider):
    provider_id: str = "mimo"
    api_key_env: str = XIAOMI_API_KEY_ENV
    base_url_env: str = XIAOMI_BASE_URL_ENV

    def generate(self, request: LLMRequest, model: LLMModelConfig) -> LLMResponse:
        api_key = _resolve_api_key(self.api_key_env)
        if not api_key:
            env_names = " or ".join(MIMO_API_KEY_ENV_ALIASES)
            raise LLMAuthError(f"{env_names} is required for MiMo/Xiaomi provider.")
        if not model.model_id.strip():
            raise LLMResponseError("MiMo model_id is required.")

        endpoint = chat_completions_endpoint(
            _resolve_base_url(self.base_url_env),
            MIMO_DEFAULT_BASE_URL,
        )
        timeout_sec = model.timeout_sec if model.timeout_sec is not None else request.timeout_sec
        parsed, latency_ms = post_chat_completions(
            provider_label="MiMo",
            endpoint=endpoint,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            payload=build_chat_payload(
                request=request,
                model=model,
                max_tokens_field="max_completion_tokens",
                include_stream_false=True,
            ),
            timeout_sec=timeout_sec,
        )
        output_text, finish_reason = extract_choice(parsed, provider_label="MiMo")
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
            request_id=str(parsed.get("id") or f"mimo-{uuid4().hex}"),
        )


def _resolve_api_key(primary_env: str) -> str:
    return _first_env_value((primary_env, *MIMO_API_KEY_ENV_ALIASES))


def _resolve_base_url(primary_env: str) -> str:
    return _first_env_value((primary_env, *MIMO_BASE_URL_ENV_ALIASES))


def _env_value(name: str) -> str:
    return os.environ.get(name, "").strip()


def _first_env_value(names: tuple[str, ...]) -> str:
    seen: set[str] = set()
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        value = _env_value(name)
        if value:
            return value
    return ""
