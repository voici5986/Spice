from __future__ import annotations

import json
import os
import time
import urllib.error as urllib_error
import urllib.request as urllib_request
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from spice.llm.core.provider import (
    LLMAuthError,
    LLMProvider,
    LLMRateLimitError,
    LLMResponseError,
    LLMTransportError,
)
from spice.llm.core.types import LLMModelConfig, LLMRequest, LLMResponse


ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
ANTHROPIC_BASE_URL_ENV = "SPICE_ANTHROPIC_BASE_URL"
ANTHROPIC_VERSION_ENV = "SPICE_ANTHROPIC_VERSION"
ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_DEFAULT_VERSION = "2023-06-01"
ANTHROPIC_DEFAULT_MAX_TOKENS = 1024


@dataclass(slots=True)
class AnthropicLLMProvider(LLMProvider):
    provider_id: str = "anthropic"
    api_key_env: str = ANTHROPIC_API_KEY_ENV
    base_url_env: str = ANTHROPIC_BASE_URL_ENV
    version_env: str = ANTHROPIC_VERSION_ENV

    def generate(self, request: LLMRequest, model: LLMModelConfig) -> LLMResponse:
        api_key = _env_value(self.api_key_env)
        if not api_key:
            raise LLMAuthError(f"{self.api_key_env} is required for Anthropic provider.")
        if not model.model_id.strip():
            raise LLMResponseError("Anthropic model_id is required.")

        endpoint = _messages_endpoint(_env_value(self.base_url_env))
        body = json.dumps(
            _build_messages_payload(request=request, model=model),
            ensure_ascii=True,
        ).encode("utf-8")
        http_request = urllib_request.Request(
            endpoint,
            data=body,
            headers=self._headers(api_key),
            method="POST",
        )

        timeout_sec = model.timeout_sec if model.timeout_sec is not None else request.timeout_sec
        start = time.perf_counter()
        try:
            with urllib_request.urlopen(http_request, timeout=timeout_sec) as response:
                response_body = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            raise _normalize_http_error(exc) from exc
        except urllib_error.URLError as exc:
            raise LLMTransportError(f"Anthropic request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise LLMTransportError(f"Anthropic request timed out after {timeout_sec}s.") from exc

        latency_ms = int((time.perf_counter() - start) * 1000)
        parsed = _parse_response_json(response_body)
        output_text = _extract_text_content(parsed)
        usage = parsed.get("usage")
        if not isinstance(usage, dict):
            usage = {}
        return LLMResponse(
            provider_id=self.provider_id,
            model_id=str(parsed.get("model") or model.model_id),
            output_text=output_text,
            raw_payload=parsed,
            finish_reason=str(parsed.get("stop_reason") or ""),
            usage=usage,
            latency_ms=latency_ms,
            request_id=str(parsed.get("id") or f"an-{uuid4().hex}"),
        )

    def _headers(self, api_key: str) -> dict[str, str]:
        return {
            "x-api-key": api_key,
            "anthropic-version": _env_value(self.version_env) or ANTHROPIC_DEFAULT_VERSION,
            "Content-Type": "application/json",
        }


def _build_messages_payload(*, request: LLMRequest, model: LLMModelConfig) -> dict[str, Any]:
    max_tokens = model.max_tokens if model.max_tokens is not None else request.max_tokens
    payload: dict[str, Any] = {
        "model": model.model_id,
        "max_tokens": max_tokens if max_tokens is not None else ANTHROPIC_DEFAULT_MAX_TOKENS,
        "messages": [{"role": "user", "content": request.input_text}],
    }
    system_text = request.system_text.strip()
    if system_text:
        payload["system"] = system_text
    if model.temperature is not None:
        payload["temperature"] = model.temperature
    return payload


def _messages_endpoint(base_url: str | None) -> str:
    normalized = (base_url or ANTHROPIC_DEFAULT_BASE_URL).strip()
    if not normalized:
        normalized = ANTHROPIC_DEFAULT_BASE_URL
    return normalized.rstrip("/") + "/messages"


def _parse_response_json(raw: str) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMResponseError("Anthropic response was not valid JSON.") from exc
    if not isinstance(payload, dict):
        raise LLMResponseError("Anthropic response JSON must be an object.")
    return payload


def _extract_text_content(payload: dict[str, Any]) -> str:
    content = payload.get("content")
    if not isinstance(content, list) or not content:
        raise LLMResponseError("Anthropic response missing content blocks.")
    text_blocks: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            text = str(block.get("text") or "").strip()
            if text:
                text_blocks.append(text)
    output = "\n".join(text_blocks).strip()
    if not output:
        raise LLMResponseError("Anthropic response content has no text blocks.")
    return output


def _normalize_http_error(exc: urllib_error.HTTPError) -> Exception:
    body = _safe_error_body(exc)
    reason = str(getattr(exc, "reason", "") or getattr(exc, "msg", "") or "")
    message = (
        "Anthropic request failed "
        f"(status={exc.code}): {body or reason or '<no response body>'}"
    )
    if exc.code in (401, 403):
        return LLMAuthError(message)
    if exc.code == 429:
        return LLMRateLimitError(message)
    if exc.code in (400, 404, 422):
        return LLMResponseError(message)
    return LLMTransportError(message)


def _safe_error_body(exc: urllib_error.HTTPError) -> str:
    try:
        body = exc.read()
    except Exception:
        return ""
    if not body:
        return ""
    try:
        return body.decode("utf-8").strip()
    except Exception:
        return repr(body)


def _env_value(name: str) -> str:
    return os.environ.get(name, "").strip()
