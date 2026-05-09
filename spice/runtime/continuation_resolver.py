from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from spice.llm.candidate_expander import build_candidate_expander_client
from spice.llm.core import LLMClient, LLMRequest, LLMTaskHook
from spice.llm.util import extract_first_json_object


@dataclass(frozen=True, slots=True)
class ContinuationResolution:
    is_continuation: bool
    action: str = "new_intent"
    candidate_id: str = ""
    label: str = ""
    text: str = ""
    reason: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "is_continuation": self.is_continuation,
            "action": self.action,
            "candidate_id": self.candidate_id,
            "label": self.label,
            "text": self.text,
            "reason": self.reason,
        }


def resolve_continuation(
    user_input: str,
    active_frame: Mapping[str, Any] | None,
) -> ContinuationResolution:
    text = user_input.strip()
    if not text or not isinstance(active_frame, Mapping) or not active_frame:
        return ContinuationResolution(False, text=text)

    normalized = _normalize(text)
    label = _option_label_from_text(normalized)
    if label:
        candidate = _candidate_by_label(active_frame, label)
        if candidate:
            return ContinuationResolution(
                True,
                action="choose_option",
                candidate_id=str(candidate.get("candidate_id") or ""),
                label=str(candidate.get("label") or label),
                text=text,
                reason="User referenced a visible Decision Card option.",
            )

    if _is_execute_selected(normalized):
        selected = _mapping(active_frame.get("selected"))
        return ContinuationResolution(
            True,
            action="execute_selected",
            candidate_id=str(selected.get("candidate_id") or active_frame.get("selected_candidate_id") or ""),
            label=str(selected.get("label") or ""),
            text=text,
            reason="User asked to execute the current selected decision.",
        )

    approval_id = str(active_frame.get("approval_id") or "").strip()
    if approval_id and normalized in {
        "y",
        "yes",
        "approve and execute",
        "approve execute",
        "approve then execute",
        "批准并执行",
    }:
        return ContinuationResolution(
            True,
            action="approve_execute",
            candidate_id=str(active_frame.get("selected_candidate_id") or ""),
            text=text,
            reason="User approved and asked to execute the pending selected decision.",
        )

    if normalized in {"approve", "approved", "yes", "y", "ok"}:
        return ContinuationResolution(
            True,
            action="approve_only",
            candidate_id=str(active_frame.get("selected_candidate_id") or ""),
            text=text,
            reason="User approved the current selected decision.",
        )

    if normalized in {"details", "detail", "why", "show details", "show why", "详情", "为什么"}:
        return ContinuationResolution(
            True,
            action="show_details",
            candidate_id=str(active_frame.get("selected_candidate_id") or ""),
            text=text,
            reason="User asked for details about the current Decision Card.",
        )

    if normalized in {"skip", "later", "not now", "先跳过", "跳过"}:
        return ContinuationResolution(
            True,
            action="skip",
            candidate_id=str(active_frame.get("selected_candidate_id") or ""),
            text=text,
            reason="User skipped the current Decision Card.",
        )

    refine_text = _refine_text(normalized, text)
    if refine_text:
        return ContinuationResolution(
            True,
            action="refine",
            candidate_id=str(active_frame.get("selected_candidate_id") or ""),
            text=refine_text,
            reason="User refined the current Decision Card.",
        )

    return ContinuationResolution(False, text=text)


def resolve_continuation_from_runtime_config(
    user_input: str,
    active_frame: Mapping[str, Any] | None,
    *,
    config: Mapping[str, Any] | None,
) -> ContinuationResolution:
    resolution = resolve_continuation(user_input, active_frame)
    if resolution.is_continuation:
        return resolution
    if not _llm_fallback_enabled(config):
        return resolution

    provider_id = str(_mapping(config).get("llm_provider") or "deterministic").strip()
    model_id = _runtime_model_id(_mapping(config))
    if not model_id:
        return resolution
    try:
        client = build_candidate_expander_client(
            provider_id=provider_id,
            model_id=model_id,
        )
        return resolve_continuation_with_llm(
            user_input,
            active_frame,
            client=client,
            model_provider=provider_id,
            model_id=model_id,
        )
    except Exception:
        return resolution


def resolve_continuation_with_llm(
    user_input: str,
    active_frame: Mapping[str, Any] | None,
    *,
    client: LLMClient,
    model_provider: str = "",
    model_id: str = "",
) -> ContinuationResolution:
    text = user_input.strip()
    if not text or not isinstance(active_frame, Mapping) or not active_frame:
        return ContinuationResolution(False, text=text)
    response = client.generate(
        LLMRequest(
            task_hook=LLMTaskHook.DECISION_PROPOSE,
            input_text=_llm_fallback_prompt(text, active_frame),
            system_text=_llm_fallback_system_prompt(),
            response_format_hint="json_object",
            temperature=0.0,
            max_tokens=500,
            timeout_sec=20.0,
            metadata={
                "purpose": "continuation_resolution",
                "model_provider": model_provider,
                "model_id": model_id,
            },
        )
    )
    payload = _parse_llm_resolution_payload(response.output_text)
    return _resolution_from_llm_payload(text, active_frame, payload)


def selected_candidate_execution_text(active_frame: Mapping[str, Any]) -> str:
    selected = _mapping(active_frame.get("selected"))
    return (
        str(selected.get("executor_task") or "").strip()
        or str(selected.get("recommended_action") or "").strip()
        or str(selected.get("intent") or "").strip()
        or str(selected.get("title") or "").strip()
    )


def update_frame_selected_candidate(
    active_frame: Mapping[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    frame = dict(active_frame)
    candidates = [_mapping(candidate) for candidate in _list(frame.get("candidates"))]
    selected = None
    updated_candidates = []
    for candidate in candidates:
        is_selected = str(candidate.get("candidate_id") or "") == candidate_id
        candidate["is_selected"] = is_selected
        if is_selected:
            selected = dict(candidate)
        updated_candidates.append(candidate)
    if selected is None:
        return frame
    frame["candidates"] = updated_candidates
    frame["selected"] = selected
    frame["selected_candidate_id"] = candidate_id
    frame["status"] = "selected"
    original_approval_id = str(frame.get("approval_id") or "")
    if original_approval_id and candidate_id != str(active_frame.get("selected_candidate_id") or ""):
        frame["approval_id"] = ""
        frame["status"] = "selected_without_approval"
    return frame


def _is_execute_selected(normalized: str) -> bool:
    return normalized in {
        "execute",
        "execute selected",
        "execute this",
        "act",
        "act on selected",
        "act on this",
        "do it",
        "make it happen",
        "run it",
        "go",
        "执行",
        "执行这个",
        "执行选中",
        "就执行这个",
    }


def _refine_text(normalized: str, original: str) -> str:
    prefixes = [
        "refine that",
        "refine this",
        "refine",
        "make it",
        "make this",
        "adjust it",
        "revise it",
        "改一下",
        "调整一下",
        "优化一下",
        "细化一下",
    ]
    for prefix in prefixes:
        if normalized == prefix:
            return "Refine the selected decision."
        if normalized.startswith(prefix + " "):
            return original[len(prefix):].strip(" :;,.") or "Refine the selected decision."
    return ""


def _option_label_from_text(normalized: str) -> str:
    exact = _label_token(normalized)
    if exact:
        return exact
    patterns = [
        r"^(?:choose|pick|select|use|go with|take|do)\s+([a-z])$",
        r"^(?:choose|pick|select|use|go with|take|do)\s+(first|second|third|1st|2nd|3rd|one|two|three)$",
        r"^选\s*([a-z])$",
        r"^选择\s*([a-z])$",
        r"^选\s*(第?一|第?二|第?三|1|2|3)个?$",
        r"^选择\s*(第?一|第?二|第?三|1|2|3)个?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, normalized)
        if not match:
            continue
        label = _label_token(match.group(1))
        if label:
            return label
    return ""


def _label_token(token: str) -> str:
    value = token.strip().lower()
    if len(value) == 1 and "a" <= value <= "z":
        return value.upper()
    aliases = {
        "first": "A",
        "1st": "A",
        "one": "A",
        "1": "A",
        "一": "A",
        "第一": "A",
        "second": "B",
        "2nd": "B",
        "two": "B",
        "2": "B",
        "二": "B",
        "第二": "B",
        "third": "C",
        "3rd": "C",
        "three": "C",
        "3": "C",
        "三": "C",
        "第三": "C",
    }
    return aliases.get(value, "")


def _candidate_by_label(active_frame: Mapping[str, Any], label: str) -> dict[str, Any] | None:
    for candidate in _list(active_frame.get("candidates")):
        item = _mapping(candidate)
        if str(item.get("label") or "").upper() == label.upper():
            return item
    return None


def _candidate_by_id(active_frame: Mapping[str, Any], candidate_id: str) -> dict[str, Any] | None:
    for candidate in _list(active_frame.get("candidates")):
        item = _mapping(candidate)
        if str(item.get("candidate_id") or "") == candidate_id:
            return item
    return None


def _normalize(text: str) -> str:
    return " ".join(text.strip().lower().split())


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _llm_fallback_enabled(config: Mapping[str, Any] | None) -> bool:
    payload = _mapping(config)
    provider_id = str(payload.get("llm_provider") or "deterministic").strip()
    if provider_id == "deterministic":
        return False
    return _truthy(payload.get("llm_candidate_expand")) or _truthy(payload.get("llm_simulation"))


def _runtime_model_id(config: Mapping[str, Any]) -> str:
    configured = str(config.get("llm_model") or "").strip()
    if configured:
        return configured
    provider_id = str(config.get("llm_provider") or "deterministic").strip()
    if provider_id == "deterministic":
        return "deterministic.v1"
    return ""


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on", "enabled"}


def _llm_fallback_system_prompt() -> str:
    return (
        "You classify whether a user message continues the active Spice Decision Card. "
        "Return only one JSON object. Do not answer the user. Do not execute. "
        "Be conservative: if the message introduces a new topic or asks a fresh question, "
        "return is_continuation=false."
    )


def _llm_fallback_prompt(user_input: str, active_frame: Mapping[str, Any]) -> str:
    payload = {
        "task": "Classify a possible continuation of the active Decision Card.",
        "allowed_actions": [
            "choose_option",
            "execute_selected",
            "approve_execute",
            "approve_only",
            "refine",
            "show_details",
            "skip",
            "new_intent",
        ],
        "rules": [
            "Use choose_option when the user picks one visible option by label, ordinal, title, or meaning.",
            "Use execute_selected when the user says to start, do it, implement it, make it happen, or equivalent.",
            "Use refine when the user asks to adjust the current Decision Card.",
            "Use new_intent for unrelated new requests.",
        ],
        "response_schema": {
            "is_continuation": "boolean",
            "action": "one allowed action",
            "candidate_id": "candidate id for choose_option if known",
            "candidate_label": "visible label like A/B/C if known",
            "refinement": "refinement text when action=refine",
            "reason": "short reason",
        },
        "user_input": user_input,
        "active_decision_frame": _compact_frame_for_llm(active_frame),
    }
    return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True)


def _compact_frame_for_llm(active_frame: Mapping[str, Any]) -> dict[str, Any]:
    selected = _mapping(active_frame.get("selected"))
    return {
        "decision_id": str(active_frame.get("decision_id") or ""),
        "selected_candidate_id": str(active_frame.get("selected_candidate_id") or ""),
        "approval_id": str(active_frame.get("approval_id") or ""),
        "selected": _compact_candidate_for_llm(selected),
        "candidates": [
            _compact_candidate_for_llm(_mapping(candidate))
            for candidate in _list(active_frame.get("candidates"))[:6]
        ],
        "allowed_continuations": _list(active_frame.get("allowed_continuations"))[:8],
    }


def _compact_candidate_for_llm(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": str(candidate.get("label") or ""),
        "candidate_id": str(candidate.get("candidate_id") or ""),
        "title": str(candidate.get("title") or ""),
        "recommendation": str(
            candidate.get("recommended_action")
            or candidate.get("recommendation")
            or candidate.get("intent")
            or ""
        ),
        "executor_task": str(candidate.get("executor_task") or ""),
        "is_selected": bool(candidate.get("is_selected")),
    }


def _parse_llm_resolution_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if not stripped:
        return {}
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        extracted = extract_first_json_object(stripped)
        payload = json.loads(extracted) if extracted else None
    return payload if isinstance(payload, dict) else {}


def _resolution_from_llm_payload(
    text: str,
    active_frame: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> ContinuationResolution:
    if not _truthy(payload.get("is_continuation")):
        return ContinuationResolution(False, text=text, reason=str(payload.get("reason") or ""))
    action = str(payload.get("action") or "").strip().lower()
    if action == "new_intent":
        return ContinuationResolution(False, text=text, reason=str(payload.get("reason") or ""))

    if action == "choose_option":
        candidate = None
        candidate_id = str(payload.get("candidate_id") or "").strip()
        if candidate_id:
            candidate = _candidate_by_id(active_frame, candidate_id)
        if candidate is None:
            label = _label_token(str(payload.get("candidate_label") or ""))
            candidate = _candidate_by_label(active_frame, label) if label else None
        if candidate is None:
            return ContinuationResolution(False, text=text, reason="LLM chose an unknown candidate.")
        return ContinuationResolution(
            True,
            action="choose_option",
            candidate_id=str(candidate.get("candidate_id") or ""),
            label=str(candidate.get("label") or payload.get("candidate_label") or ""),
            text=text,
            reason=str(payload.get("reason") or "LLM matched the message to a visible option."),
        )

    selected = _mapping(active_frame.get("selected"))
    selected_id = str(selected.get("candidate_id") or active_frame.get("selected_candidate_id") or "")
    if action in {"execute_selected", "approve_execute", "approve_only", "show_details", "skip"}:
        return ContinuationResolution(
            True,
            action=action,
            candidate_id=selected_id,
            label=str(selected.get("label") or ""),
            text=text,
            reason=str(payload.get("reason") or "LLM classified this as a continuation."),
        )
    if action == "refine":
        refinement = str(payload.get("refinement") or text).strip()
        return ContinuationResolution(
            True,
            action="refine",
            candidate_id=selected_id,
            label=str(selected.get("label") or ""),
            text=refinement,
            reason=str(payload.get("reason") or "LLM classified this as refinement."),
        )
    return ContinuationResolution(False, text=text, reason=f"Unsupported LLM continuation action: {action}")
