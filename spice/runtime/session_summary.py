from __future__ import annotations

import json
from typing import Any

from spice.decision.general.types import payload_value
from spice.llm.core import LLMClient, LLMRequest, LLMTaskHook
from spice.llm.util import extract_first_json_object
from spice.memory import FileMemoryProvider, MemoryProvider
from spice.protocols import utc_now

GENERAL_SESSION_SUMMARY_NAMESPACE = "general.session_summary"
GENERAL_SESSION_SUMMARY_SCHEMA_VERSION = "spice.memory.general.session_summary.v1"
GENERAL_LLM_SESSION_SUMMARY_SCHEMA_VERSION = "spice.memory.general.session_summary.llm.v1"
DEFAULT_MEMORY_SUMMARY_LLM_MIN_NEW_RECORDS = 4
DEFAULT_MEMORY_SUMMARY_TRIGGER_CHARS = 8000
DEFAULT_MEMORY_SUMMARY_TARGET_CHARS = 6000


def update_session_summary(
    provider: MemoryProvider,
    *,
    config: dict[str, Any] | None = None,
    llm_client: LLMClient | None = None,
    domain: str = "general",
    recent_decision_limit: int = 5,
    recent_reflection_limit: int = 5,
    preference_limit: int = 5,
) -> dict[str, Any]:
    """Update deterministic summary and optionally roll it into an LLM summary."""

    deterministic = update_deterministic_session_summary(
        provider,
        domain=domain,
        recent_decision_limit=recent_decision_limit,
        recent_reflection_limit=recent_reflection_limit,
        preference_limit=preference_limit,
    )
    config = dict(config or {})
    if str(config.get("memory_summary_provider") or "deterministic") != "llm":
        return {
            **deterministic,
            "summary_source": "deterministic",
            "llm_summary": {
                "enabled": False,
                "status": "skipped",
                "reason": "memory_summary_provider is deterministic",
            },
        }

    deterministic_record = latest_session_summary(provider, domain=domain)
    llm_result = maybe_update_llm_session_summary(
        provider,
        deterministic_record=deterministic_record,
        config=config,
        llm_client=llm_client,
        domain=domain,
    )
    source = "llm" if llm_result.get("status") == "written" else "deterministic"
    return {
        **deterministic,
        "summary_source": source,
        "llm_summary": llm_result,
    }


def update_deterministic_session_summary(
    provider: MemoryProvider,
    *,
    domain: str = "general",
    recent_decision_limit: int = 5,
    recent_reflection_limit: int = 5,
    preference_limit: int = 5,
) -> dict[str, Any]:
    """Regenerate the latest deterministic decision-relevant session summary."""

    namespace = f"{domain}.session_summary"
    record = build_deterministic_session_summary(
        provider,
        domain=domain,
        recent_decision_limit=recent_decision_limit,
        recent_reflection_limit=recent_reflection_limit,
        preference_limit=preference_limit,
    )
    record_ids = provider.write([record], namespace=namespace, refs=_summary_refs(record))
    latest_files = _write_latest_summary_files(provider, record)
    return {
        "enabled": True,
        "status": "written",
        "namespace": namespace,
        "record_ids": record_ids,
        "refs": _summary_refs(record),
        "latest_files": latest_files,
    }


def build_deterministic_session_summary(
    provider: MemoryProvider,
    *,
    domain: str = "general",
    recent_decision_limit: int = 5,
    recent_reflection_limit: int = 5,
    preference_limit: int = 5,
) -> dict[str, Any]:
    decisions = _records(provider, f"{domain}.decision")
    reflections = _records(provider, f"{domain}.reflection")
    preferences = _records(provider, f"{domain}.preference")
    recent_decisions = [_compact_decision(record) for record in decisions[-recent_decision_limit:]]
    execution_outcomes = [
        _compact_reflection(record) for record in reflections[-recent_reflection_limit:]
    ]
    user_preferences = [
        _compact_preference(record) for record in preferences[-preference_limit:]
    ]
    latest_decision = decisions[-1] if decisions else {}
    latest_reflection = reflections[-1] if reflections else {}
    current_goal = _current_goal(latest_decision)
    active_decision = _active_decision(latest_decision)
    open_threads = _open_threads(
        latest_decision=latest_decision,
        latest_reflection=latest_reflection,
        recent_decisions=recent_decisions,
        execution_outcomes=execution_outcomes,
    )
    record = {
        "id": "memory.general.session_summary.generated",
        "schema_version": GENERAL_SESSION_SUMMARY_SCHEMA_VERSION,
        "record_type": "general.session_summary",
        "summary_type": "deterministic",
        "domain": domain,
        "updated_at": utc_now().isoformat(),
        "source_namespaces": [
            f"{domain}.decision",
            f"{domain}.reflection",
            f"{domain}.preference",
        ],
        "counts": {
            "decisions": len(decisions),
            "reflections": len(reflections),
            "preferences": len(preferences),
        },
        "covered_record_ids": _covered_record_ids(
            decisions=decisions,
            reflections=reflections,
            preferences=preferences,
        ),
        "current_goal": current_goal,
        "active_decision": active_decision,
        "user_preferences": user_preferences,
        "recent_decisions": recent_decisions,
        "execution_outcomes": execution_outcomes,
        "open_threads": open_threads,
    }
    record["markdown"] = render_session_summary_markdown(record)
    return record


def maybe_update_llm_session_summary(
    provider: MemoryProvider,
    *,
    deterministic_record: dict[str, Any],
    config: dict[str, Any],
    llm_client: LLMClient | None = None,
    domain: str = "general",
) -> dict[str, Any]:
    provider_id = str(config.get("llm_provider") or "deterministic").strip()
    model_id = str(config.get("llm_model") or "").strip()
    if provider_id == "deterministic" or not model_id:
        return {
            "enabled": True,
            "status": "fallback",
            "reason": "LLM summary requested but no non-deterministic LLM model is configured.",
        }

    trigger = _llm_summary_trigger(provider, deterministic_record, config=config, domain=domain)
    if not trigger["triggered"]:
        return {
            "enabled": True,
            "status": "skipped",
            "reason": trigger["reason"],
            "new_record_count": trigger["new_record_count"],
            "estimated_chars": trigger["estimated_chars"],
        }

    previous_llm = _latest_llm_summary(provider, domain=domain)
    prompt = _build_llm_summary_prompt(
        deterministic_record=deterministic_record,
        previous_llm_summary=previous_llm,
        recent_records=_recent_raw_records(provider, domain=domain),
        target_chars=_positive_int(
            config.get("memory_summary_target_chars"),
            DEFAULT_MEMORY_SUMMARY_TARGET_CHARS,
        ),
    )
    try:
        client = llm_client or _build_llm_summary_client(
            provider_id=provider_id,
            model_id=model_id,
        )
        response = client.generate(
            LLMRequest(
                task_hook=LLMTaskHook.SESSION_SUMMARIZE,
                domain=domain,
                input_text=prompt,
                system_text=_llm_summary_system_prompt(),
                response_format_hint="json_object",
                temperature=0.1,
                max_tokens=1600,
                metadata={"component": "runtime.session_summary"},
            )
        )
        summary_payload = _parse_llm_summary_payload(response.output_text)
        record = _build_llm_summary_record(
            summary_payload,
            deterministic_record=deterministic_record,
            previous_llm_summary=previous_llm,
            provider_id=response.provider_id or provider_id,
            model_id=response.model_id or model_id,
            request_id=response.request_id,
            domain=domain,
        )
        _validate_llm_summary_record(record, deterministic_record)
        namespace = f"{domain}.session_summary"
        record_ids = provider.write([record], namespace=namespace, refs=_summary_refs(record))
        latest_files = _write_latest_summary_files(provider, record)
        return {
            "enabled": True,
            "status": "written",
            "namespace": namespace,
            "record_ids": record_ids,
            "refs": _summary_refs(record),
            "latest_files": latest_files,
            "summary_type": "llm",
            "reason": trigger["reason"],
            "new_record_count": trigger["new_record_count"],
            "estimated_chars": trigger["estimated_chars"],
            "request_id": response.request_id,
            "model_provider": response.provider_id or provider_id,
            "model_id": response.model_id or model_id,
        }
    except Exception as exc:  # pragma: no cover - exercised through fallback behavior
        return {
            "enabled": True,
            "status": "fallback",
            "reason": f"LLM session summary failed; deterministic summary remains active: {exc}",
        }


def render_session_summary_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Session Summary",
        "",
        "## Current Goal",
        f"- {_text(_dict(summary.get('current_goal')).get('text'), 'No current goal recorded.')}",
        "",
        "## Active Decision",
    ]
    active = _dict(summary.get("active_decision"))
    lines.append(
        "- "
        + _text(
            active.get("recommendation") or active.get("title"),
            "No active decision recorded.",
        )
    )
    if active.get("decision_id"):
        lines.append(f"- decision_id: {active['decision_id']}")
    if active.get("selected_candidate_id"):
        lines.append(f"- selected_candidate_id: {active['selected_candidate_id']}")
    if active.get("approval_id"):
        lines.append(f"- approval_id: {active['approval_id']}")

    lines.extend(["", "## User Preferences"])
    preferences = _list(summary.get("user_preferences"))
    if preferences:
        lines.extend(f"- {_text(item.get('summary'), item.get('id', 'preference'))}" for item in preferences)
    else:
        lines.append("- No explicit preferences recorded.")

    lines.extend(["", "## Recent Decisions"])
    decisions = _list(summary.get("recent_decisions"))
    if decisions:
        for item in decisions:
            title = _text(item.get("recommendation") or item.get("title"), "Decision recorded.")
            decision_id = _text(item.get("decision_id"), "")
            lines.append(f"- {title}" + (f" ({decision_id})" if decision_id else ""))
    else:
        lines.append("- No prior decisions recorded.")

    lines.extend(["", "## Execution Outcomes"])
    outcomes = _list(summary.get("execution_outcomes"))
    if outcomes:
        for item in outcomes:
            status = _text(item.get("task_status"), "unknown")
            executor = _text(item.get("executor"), "executor")
            title = _text(item.get("selected_title") or item.get("summary"), "Execution recorded.")
            lines.append(f"- {executor}: {status} - {title}")
    else:
        lines.append("- No execution outcomes recorded.")

    lines.extend(["", "## Open Threads"])
    threads = _list(summary.get("open_threads"))
    if threads:
        for item in threads:
            lines.append(f"- {_text(item.get('summary'), item.get('kind', 'open thread'))}")
    else:
        lines.append("- No open decision threads recorded.")

    return "\n".join(lines).strip() + "\n"


def latest_session_summary(provider: MemoryProvider, *, domain: str = "general") -> dict[str, Any]:
    records = _records(provider, f"{domain}.session_summary")
    return records[-1] if records else {}


def compact_session_summary_record(record: dict[str, Any]) -> dict[str, Any]:
    if not record:
        return {}
    return {
        "id": str(record.get("id") or ""),
        "schema_version": str(record.get("schema_version") or ""),
        "summary_type": str(record.get("summary_type") or ""),
        "domain": str(record.get("domain") or ""),
        "updated_at": str(record.get("updated_at") or ""),
        "counts": _dict(record.get("counts")),
        "covered_record_ids": _list(record.get("covered_record_ids")),
        "model": _dict(record.get("model")),
        "current_goal": _dict(record.get("current_goal")),
        "active_decision": _dict(record.get("active_decision")),
        "user_preferences": _list(record.get("user_preferences")),
        "recent_decisions": _list(record.get("recent_decisions")),
        "execution_outcomes": _list(record.get("execution_outcomes")),
        "open_threads": _list(record.get("open_threads")),
        "markdown": str(record.get("markdown") or ""),
    }


def _records(provider: MemoryProvider, namespace: str) -> list[dict[str, Any]]:
    return provider.query(namespace=namespace, limit=-1)


def _covered_record_ids(
    *,
    decisions: list[dict[str, Any]],
    reflections: list[dict[str, Any]],
    preferences: list[dict[str, Any]],
) -> list[str]:
    ids: list[str] = []
    for record in [*decisions, *reflections, *preferences]:
        record_id = str(record.get("id") or "")
        if record_id:
            ids.append(record_id)
    return list(dict.fromkeys(ids))


def _latest_llm_summary(provider: MemoryProvider, *, domain: str) -> dict[str, Any]:
    records = _records(provider, f"{domain}.session_summary")
    for record in reversed(records):
        if str(record.get("summary_type") or "") == "llm":
            return record
    return {}


def _llm_summary_trigger(
    provider: MemoryProvider,
    deterministic_record: dict[str, Any],
    *,
    config: dict[str, Any],
    domain: str,
) -> dict[str, Any]:
    latest_llm = _latest_llm_summary(provider, domain=domain)
    covered = set(str(item) for item in _list(latest_llm.get("covered_record_ids")))
    all_ids = [str(item) for item in _list(deterministic_record.get("covered_record_ids"))]
    new_record_ids = [item for item in all_ids if item and item not in covered]
    estimated_chars = len(json.dumps(deterministic_record, ensure_ascii=False, sort_keys=True))
    min_new_records = _positive_int(
        config.get("memory_summary_llm_min_new_records"),
        DEFAULT_MEMORY_SUMMARY_LLM_MIN_NEW_RECORDS,
    )
    trigger_chars = _positive_int(
        config.get("memory_summary_trigger_chars"),
        DEFAULT_MEMORY_SUMMARY_TRIGGER_CHARS,
    )
    important = _has_recent_failed_execution(provider, latest_llm=latest_llm, domain=domain)
    if len(new_record_ids) >= min_new_records:
        return {
            "triggered": True,
            "reason": f"{len(new_record_ids)} new memory records reached threshold {min_new_records}",
            "new_record_count": len(new_record_ids),
            "estimated_chars": estimated_chars,
        }
    if estimated_chars >= trigger_chars:
        return {
            "triggered": True,
            "reason": f"deterministic summary reached {estimated_chars} chars",
            "new_record_count": len(new_record_ids),
            "estimated_chars": estimated_chars,
        }
    if important:
        return {
            "triggered": True,
            "reason": "recent failed execution should be preserved in rolling summary",
            "new_record_count": len(new_record_ids),
            "estimated_chars": estimated_chars,
        }
    return {
        "triggered": False,
        "reason": (
            f"waiting for {min_new_records} new records or {trigger_chars} chars "
            f"({len(new_record_ids)} new records, {estimated_chars} chars)"
        ),
        "new_record_count": len(new_record_ids),
        "estimated_chars": estimated_chars,
    }


def _has_recent_failed_execution(
    provider: MemoryProvider,
    *,
    latest_llm: dict[str, Any],
    domain: str,
) -> bool:
    covered = set(str(item) for item in _list(latest_llm.get("covered_record_ids")))
    for record in _records(provider, f"{domain}.reflection"):
        record_id = str(record.get("id") or "")
        if record_id in covered:
            continue
        if _compact_reflection(record).get("task_status") == "failed":
            return True
    return False


def _recent_raw_records(provider: MemoryProvider, *, domain: str) -> dict[str, Any]:
    return {
        "decisions": _records(provider, f"{domain}.decision")[-8:],
        "reflections": _records(provider, f"{domain}.reflection")[-8:],
        "preferences": _records(provider, f"{domain}.preference")[-8:],
    }


def _build_llm_summary_client(*, provider_id: str, model_id: str) -> LLMClient:
    from spice.llm.candidate_expander import build_candidate_expander_client

    return build_candidate_expander_client(provider_id=provider_id, model_id=model_id)


def _llm_summary_system_prompt() -> str:
    return (
        "You compact Spice decision memory into a durable decision-relevant session summary. "
        "Return only a JSON object. Preserve stable goals, selected decisions, pending threads, "
        "execution lessons, and user preferences. Drop raw logs, repeated details, and obsolete "
        "intermediate wording. Do not invent facts. Use only evidence in the input."
    )


def _build_llm_summary_prompt(
    *,
    deterministic_record: dict[str, Any],
    previous_llm_summary: dict[str, Any],
    recent_records: dict[str, Any],
    target_chars: int,
) -> str:
    payload = {
        "task": "Create a compact rolling session summary for future Spice decisions.",
        "target_chars": target_chars,
        "required_json_shape": {
            "current_goal": {"text": "string", "decision_id": "string", "run_id": "string"},
            "active_decision": {
                "decision_id": "string",
                "selected_candidate_id": "string",
                "title": "string",
                "recommendation": "string",
                "approval_id": "string",
            },
            "user_preferences": [{"summary": "string", "source": "string"}],
            "recent_decisions": [
                {
                    "decision_id": "string",
                    "title": "string",
                    "recommendation": "string",
                    "evidence_refs": ["record ids"],
                }
            ],
            "execution_outcomes": [
                {
                    "decision_id": "string",
                    "executor": "string",
                    "task_status": "success|failed|partial|unknown",
                    "summary": "string",
                    "evidence_refs": ["record ids"],
                }
            ],
            "open_threads": [{"kind": "string", "summary": "string", "evidence_refs": ["record ids"]}],
            "dropped_details": ["string"],
        },
        "previous_llm_summary": compact_session_summary_record(previous_llm_summary),
        "deterministic_summary": compact_session_summary_record(deterministic_record),
        "recent_raw_records": payload_value(recent_records),
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _parse_llm_summary_payload(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        extracted = extract_first_json_object(stripped)
        payload = json.loads(extracted) if extracted else None
    if not isinstance(payload, dict):
        raise ValueError("LLM session summary response must be a JSON object.")
    return payload


def _build_llm_summary_record(
    payload: dict[str, Any],
    *,
    deterministic_record: dict[str, Any],
    previous_llm_summary: dict[str, Any],
    provider_id: str,
    model_id: str,
    request_id: str,
    domain: str,
) -> dict[str, Any]:
    record = {
        "id": "memory.general.session_summary.llm",
        "schema_version": GENERAL_LLM_SESSION_SUMMARY_SCHEMA_VERSION,
        "record_type": "general.session_summary",
        "summary_type": "llm",
        "domain": domain,
        "updated_at": utc_now().isoformat(),
        "source_namespaces": _list(deterministic_record.get("source_namespaces")),
        "counts": _dict(deterministic_record.get("counts")),
        "covered_record_ids": _list(deterministic_record.get("covered_record_ids")),
        "previous_summary_id": str(previous_llm_summary.get("id") or ""),
        "model": {
            "provider_id": provider_id,
            "model_id": model_id,
            "request_id": request_id,
        },
        "current_goal": _dict(payload.get("current_goal"))
        or _dict(deterministic_record.get("current_goal")),
        "active_decision": _dict(payload.get("active_decision"))
        or _dict(deterministic_record.get("active_decision")),
        "user_preferences": _bounded_dict_list(payload.get("user_preferences"), 8),
        "recent_decisions": _bounded_dict_list(payload.get("recent_decisions"), 8),
        "execution_outcomes": _bounded_dict_list(payload.get("execution_outcomes"), 8),
        "open_threads": _bounded_dict_list(payload.get("open_threads"), 8),
        "dropped_details": _bounded_string_list(payload.get("dropped_details"), 12),
    }
    record["markdown"] = render_session_summary_markdown(record)
    return record


def _validate_llm_summary_record(
    record: dict[str, Any],
    deterministic_record: dict[str, Any],
) -> None:
    known_refs = _known_summary_refs(deterministic_record)
    for ref in _collect_evidence_refs(record):
        if ref and ref not in known_refs:
            raise ValueError(f"LLM summary referenced unknown evidence ref: {ref}")


def _known_summary_refs(record: dict[str, Any]) -> set[str]:
    refs = set(str(item) for item in _list(record.get("covered_record_ids")))
    refs.update(_summary_refs(record))
    for section in (
        "current_goal",
        "active_decision",
        "recent_decisions",
        "execution_outcomes",
        "open_threads",
    ):
        refs.update(_collect_identity_refs(record.get(section)))
    return {ref for ref in refs if ref}


def _collect_identity_refs(value: Any) -> list[str]:
    refs: list[str] = []
    identity_keys = {
        "id",
        "decision_id",
        "run_id",
        "trace_ref",
        "approval_id",
        "candidate_id",
        "outcome_id",
        "active_decision_frame_ref",
    }
    if isinstance(value, dict):
        for key, item in value.items():
            if key in identity_keys and str(item or ""):
                refs.append(str(item))
            else:
                refs.extend(_collect_identity_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.extend(_collect_identity_refs(item))
    return refs


def _collect_evidence_refs(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"refs", "evidence_refs"} and isinstance(item, list):
                refs.extend(str(ref) for ref in item if str(ref or ""))
            else:
                refs.extend(_collect_evidence_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.extend(_collect_evidence_refs(item))
    return refs


def _bounded_dict_list(value: Any, limit: int) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [payload_value(item) for item in value if isinstance(item, dict)][:limit]


def _bounded_string_list(value: Any, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item or "")][:limit]


def _positive_int(value: Any, fallback: int) -> int:
    try:
        parsed = int(str(value or "").strip())
    except ValueError:
        return fallback
    return parsed if parsed > 0 else fallback


def _compact_decision(record: dict[str, Any]) -> dict[str, Any]:
    selected = _dict(record.get("selected"))
    return {
        "id": str(record.get("id") or ""),
        "created_at": str(record.get("created_at") or ""),
        "session_id": str(record.get("session_id") or ""),
        "run_id": str(record.get("run_id") or ""),
        "decision_id": str(record.get("decision_id") or ""),
        "trace_ref": str(record.get("trace_ref") or ""),
        "run_intent_mode": str(record.get("run_intent_mode") or ""),
        "input": _dict(record.get("input")),
        "selected_candidate_id": str(selected.get("candidate_id") or ""),
        "title": str(selected.get("title") or ""),
        "recommendation": str(selected.get("recommendation") or ""),
        "score": selected.get("score"),
        "approval_id": str(record.get("approval_id") or ""),
        "handoff": _dict(record.get("handoff")),
        "active_decision_frame_ref": str(record.get("active_decision_frame_ref") or ""),
    }


def _compact_reflection(record: dict[str, Any]) -> dict[str, Any]:
    executor = _dict(record.get("executor"))
    execution = _dict(record.get("execution"))
    selected = _dict(record.get("selected_candidate"))
    return {
        "id": str(record.get("id") or ""),
        "created_at": str(record.get("created_at") or ""),
        "session_id": str(record.get("session_id") or ""),
        "run_id": str(record.get("run_id") or ""),
        "decision_id": str(record.get("decision_id") or ""),
        "approval_id": str(record.get("approval_id") or ""),
        "candidate_id": str(record.get("candidate_id") or ""),
        "selected_title": str(selected.get("title") or ""),
        "executor": str(executor.get("provider") or executor.get("executor_id") or ""),
        "task_status": str(execution.get("task_status") or ""),
        "protocol_status": str(execution.get("protocol_status") or ""),
        "success": bool(execution.get("success")),
        "state_updated": bool(execution.get("state_updated")),
        "outcome_id": str(execution.get("outcome_id") or ""),
        "summary": str(_dict(record.get("outcome_summary")).get("summary") or ""),
    }


def _compact_preference(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(record.get("id") or ""),
        "created_at": str(record.get("created_at") or ""),
        "summary": str(record.get("summary") or record.get("text") or ""),
        "source": str(record.get("source") or ""),
    }


def _current_goal(latest_decision: dict[str, Any]) -> dict[str, Any]:
    intent = _dict(latest_decision.get("input"))
    return {
        "text": str(intent.get("text") or ""),
        "source": str(intent.get("source") or latest_decision.get("source") or ""),
        "decision_id": str(latest_decision.get("decision_id") or ""),
        "run_id": str(latest_decision.get("run_id") or ""),
    }


def _active_decision(latest_decision: dict[str, Any]) -> dict[str, Any]:
    selected = _dict(latest_decision.get("selected"))
    frame = _dict(latest_decision.get("active_decision_frame"))
    return {
        "decision_id": str(latest_decision.get("decision_id") or frame.get("decision_id") or ""),
        "run_id": str(latest_decision.get("run_id") or frame.get("run_id") or ""),
        "selected_candidate_id": str(
            selected.get("candidate_id") or frame.get("selected_candidate_id") or ""
        ),
        "title": str(selected.get("title") or ""),
        "recommendation": str(selected.get("recommendation") or ""),
        "approval_id": str(latest_decision.get("approval_id") or frame.get("approval_id") or ""),
        "run_intent_mode": str(latest_decision.get("run_intent_mode") or ""),
        "handoff": _dict(latest_decision.get("handoff")),
    }


def _open_threads(
    *,
    latest_decision: dict[str, Any],
    latest_reflection: dict[str, Any],
    recent_decisions: list[dict[str, Any]],
    execution_outcomes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    threads: list[dict[str, Any]] = []
    latest = recent_decisions[-1] if recent_decisions else _compact_decision(latest_decision)
    latest_reflection_compact = _compact_reflection(latest_reflection) if latest_reflection else {}
    latest_approval_handled = bool(
        latest.get("approval_id")
        and latest_reflection_compact.get("approval_id") == latest.get("approval_id")
    )
    if (
        latest.get("approval_id")
        and _dict(latest.get("handoff")).get("required")
        and not latest_approval_handled
    ):
        threads.append(
            {
                "kind": "approval",
                "decision_id": latest.get("decision_id", ""),
                "approval_id": latest.get("approval_id", ""),
                "summary": f"Approval available for {latest.get('title') or latest.get('recommendation') or latest.get('decision_id')}.",
            }
        )
    failed = [item for item in execution_outcomes if item.get("task_status") == "failed"]
    if failed:
        last_failed = failed[-1]
        threads.append(
            {
                "kind": "failed_execution",
                "decision_id": last_failed.get("decision_id", ""),
                "outcome_id": last_failed.get("outcome_id", ""),
                "summary": f"Last failed execution used {last_failed.get('executor') or 'executor'} for {last_failed.get('selected_title') or last_failed.get('candidate_id')}.",
            }
        )
    if latest_decision and not latest_reflection:
        active = _active_decision(latest_decision)
        threads.append(
            {
                "kind": "active_decision",
                "decision_id": active.get("decision_id", ""),
                "summary": f"Active decision is {active.get('title') or active.get('recommendation') or active.get('decision_id')}.",
            }
        )
    return _dedupe_threads(threads)


def _summary_refs(record: dict[str, Any]) -> list[str]:
    refs: list[str] = []
    active = _dict(record.get("active_decision"))
    for key in ("decision_id", "run_id", "approval_id"):
        value = active.get(key)
        if value:
            refs.append(str(value))
    for item in _list(record.get("execution_outcomes")):
        for key in ("outcome_id", "decision_id", "run_id", "approval_id"):
            value = item.get(key)
            if value:
                refs.append(str(value))
    return list(dict.fromkeys(refs))


def _write_latest_summary_files(
    provider: MemoryProvider,
    record: dict[str, Any],
) -> dict[str, str]:
    if not isinstance(provider, FileMemoryProvider):
        return {}
    json_path = provider.base_dir / "session_summary.generated.json"
    markdown_path = provider.base_dir / "session_summary.md"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(str(record.get("markdown") or ""), encoding="utf-8")
    return {
        "generated_json": str(json_path),
        "markdown": str(markdown_path),
    }


def _dedupe_threads(threads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for thread in threads:
        key = (str(thread.get("kind") or ""), str(thread.get("summary") or ""))
        if key in seen:
            continue
        seen.add(key)
        result.append(thread)
    return result


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _text(value: Any, fallback: str) -> str:
    text = str(value or "").strip()
    return text or fallback
