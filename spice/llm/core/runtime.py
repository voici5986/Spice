from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Mapping

from spice.llm.providers.anthropic import (
    ANTHROPIC_API_KEY_ENV,
    ANTHROPIC_BASE_URL_ENV,
    ANTHROPIC_DEFAULT_BASE_URL,
)
from spice.llm.providers.deepseek import (
    DEEPSEEK_API_KEY_ENV,
    DEEPSEEK_BASE_URL_ENV,
    DEEPSEEK_DEFAULT_BASE_URL,
)
from spice.llm.providers.mimo import (
    MIMO_API_KEY_ENV,
    MIMO_BASE_URL_ENV,
    MIMO_DEFAULT_BASE_URL,
    XIAOMI_API_KEY_ENV,
    XIAOMI_BASE_URL_ENV,
)
from spice.llm.providers.openai import (
    OPENAI_API_KEY_ENV,
    OPENAI_BASE_URL_ENV,
    OPENAI_DEFAULT_BASE_URL,
)
from spice.llm.providers.openrouter import (
    OPENROUTER_API_KEY_ENV,
    OPENROUTER_BASE_URL_ENV,
    OPENROUTER_DEFAULT_BASE_URL,
)


@dataclass(frozen=True, slots=True)
class LLMRuntimeProviderSpec:
    provider_id: str
    provider_family: str
    api_mode: str
    default_base_url: str
    api_key_envs: tuple[str, ...] = ()
    base_url_envs: tuple[str, ...] = ()
    auth_required: bool = True
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ResolvedLLMRuntime:
    requested_provider_id: str
    provider_id: str
    provider_family: str
    api_mode: str
    model_id: str
    base_url: str
    base_url_env: str
    base_url_source: str
    api_key_env: str
    api_key_present: bool
    api_key_source: str
    auth_required: bool
    status: str
    detail: str
    metadata: dict[str, str] = field(default_factory=dict)

    def to_payload(self) -> dict[str, object]:
        return {
            "requested_provider_id": self.requested_provider_id,
            "provider_id": self.provider_id,
            "provider_family": self.provider_family,
            "api_mode": self.api_mode,
            "model_id": self.model_id,
            "base_url": self.base_url,
            "base_url_env": self.base_url_env,
            "base_url_source": self.base_url_source,
            "api_key_env": self.api_key_env,
            "api_key_present": self.api_key_present,
            "api_key_source": self.api_key_source,
            "auth_required": self.auth_required,
            "status": self.status,
            "detail": self.detail,
            "metadata": dict(self.metadata),
        }


LLM_RUNTIME_PROVIDER_SPECS: Mapping[str, LLMRuntimeProviderSpec] = {
    "anthropic": LLMRuntimeProviderSpec(
        provider_id="anthropic",
        provider_family="anthropic",
        api_mode="anthropic_messages",
        default_base_url=ANTHROPIC_DEFAULT_BASE_URL,
        api_key_envs=(ANTHROPIC_API_KEY_ENV,),
        base_url_envs=(ANTHROPIC_BASE_URL_ENV,),
    ),
    "deepseek": LLMRuntimeProviderSpec(
        provider_id="deepseek",
        provider_family="openai_compatible",
        api_mode="chat_completions",
        default_base_url=DEEPSEEK_DEFAULT_BASE_URL,
        api_key_envs=(DEEPSEEK_API_KEY_ENV,),
        base_url_envs=(DEEPSEEK_BASE_URL_ENV,),
    ),
    "deterministic": LLMRuntimeProviderSpec(
        provider_id="deterministic",
        provider_family="local",
        api_mode="deterministic",
        default_base_url="",
        auth_required=False,
    ),
    "mimo": LLMRuntimeProviderSpec(
        provider_id="mimo",
        provider_family="openai_compatible",
        api_mode="chat_completions",
        default_base_url=MIMO_DEFAULT_BASE_URL,
        api_key_envs=(XIAOMI_API_KEY_ENV, MIMO_API_KEY_ENV),
        base_url_envs=(XIAOMI_BASE_URL_ENV, MIMO_BASE_URL_ENV),
        metadata={"vendor": "xiaomi"},
    ),
    "openai": LLMRuntimeProviderSpec(
        provider_id="openai",
        provider_family="openai_compatible",
        api_mode="chat_completions",
        default_base_url=OPENAI_DEFAULT_BASE_URL,
        api_key_envs=(OPENAI_API_KEY_ENV,),
        base_url_envs=(OPENAI_BASE_URL_ENV,),
    ),
    "openrouter": LLMRuntimeProviderSpec(
        provider_id="openrouter",
        provider_family="openrouter",
        api_mode="chat_completions",
        default_base_url=OPENROUTER_DEFAULT_BASE_URL,
        api_key_envs=(OPENROUTER_API_KEY_ENV,),
        base_url_envs=(OPENROUTER_BASE_URL_ENV,),
    ),
    "subprocess": LLMRuntimeProviderSpec(
        provider_id="subprocess",
        provider_family="local",
        api_mode="subprocess",
        default_base_url="",
        auth_required=False,
    ),
}

LLM_PROVIDER_ALIASES: Mapping[str, str] = {
    "xiaomi": "mimo",
    "xiaomi_mimo": "mimo",
}


def resolve_llm_runtime(
    *,
    provider_id: str,
    model_id: str = "",
    configured_api_key_env: str = "",
    env: Mapping[str, str] | None = None,
) -> ResolvedLLMRuntime:
    requested = provider_id.strip()
    normalized = _normalize_provider_id(requested)
    runtime_env = env if env is not None else os.environ
    spec = LLM_RUNTIME_PROVIDER_SPECS.get(normalized)
    if spec is None:
        return ResolvedLLMRuntime(
            requested_provider_id=requested,
            provider_id=normalized,
            provider_family="unknown",
            api_mode="unknown",
            model_id=model_id.strip(),
            base_url="",
            base_url_env="",
            base_url_source="unsupported",
            api_key_env="",
            api_key_present=False,
            api_key_source="unsupported",
            auth_required=True,
            status="unsupported",
            detail=f"Unsupported LLM provider: {requested or '<empty>'}",
        )

    api_envs = _dedupe((configured_api_key_env.strip(), *spec.api_key_envs))
    api_key_env, api_key_present = _first_present_env(api_envs, runtime_env)
    if not api_key_env and api_envs:
        api_key_env = api_envs[0]
    base_url_env, base_url = _first_env_value(spec.base_url_envs, runtime_env)
    base_url_source = base_url_env or ("default" if spec.default_base_url else "")
    if not base_url:
        base_url = spec.default_base_url

    status, detail = _runtime_status(
        spec=spec,
        model_id=model_id.strip(),
        api_key_present=api_key_present,
        api_envs=api_envs,
    )
    return ResolvedLLMRuntime(
        requested_provider_id=requested,
        provider_id=spec.provider_id,
        provider_family=spec.provider_family,
        api_mode=spec.api_mode,
        model_id=model_id.strip(),
        base_url=base_url,
        base_url_env=base_url_env,
        base_url_source=base_url_source,
        api_key_env=api_key_env,
        api_key_present=api_key_present,
        api_key_source=api_key_env if api_key_present else "",
        auth_required=spec.auth_required,
        status=status,
        detail=detail,
        metadata=dict(spec.metadata),
    )


def _normalize_provider_id(provider_id: str) -> str:
    token = provider_id.strip().lower().replace("-", "_")
    return LLM_PROVIDER_ALIASES.get(token, token)


def _runtime_status(
    *,
    spec: LLMRuntimeProviderSpec,
    model_id: str,
    api_key_present: bool,
    api_envs: tuple[str, ...],
) -> tuple[str, str]:
    if spec.provider_id == "deterministic":
        return "ready", "deterministic provider does not require credentials"
    if spec.provider_id == "subprocess":
        if model_id:
            return "ready", "subprocess command configured as model_id"
        return "warning", "subprocess provider requires model_id command"
    if not model_id:
        return "warning", "model_id is missing"
    if spec.auth_required and not api_key_present:
        env_label = " or ".join(api_envs) if api_envs else "provider API key"
        return "warning", f"{env_label} is not set"
    return "ready", "provider runtime resolved"


def _first_present_env(
    names: tuple[str, ...],
    env: Mapping[str, str],
) -> tuple[str, bool]:
    for name in names:
        if not name:
            continue
        value = str(env.get(name, "") or "").strip()
        if value:
            return name, True
    return "", False


def _first_env_value(
    names: tuple[str, ...],
    env: Mapping[str, str],
) -> tuple[str, str]:
    for name in names:
        if not name:
            continue
        value = str(env.get(name, "") or "").strip()
        if value:
            return name, value
    return "", ""


def _dedupe(names: tuple[str, ...]) -> tuple[str, ...]:
    result: list[str] = []
    for name in names:
        normalized = name.strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return tuple(result)
