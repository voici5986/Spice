from __future__ import annotations

from typing import Any

from spice.decision.general.types import payload_value
from spice.memory import MemoryProvider
from spice.runtime.session_summary import update_session_summary


GENERAL_DECISION_MEMORY_NAMESPACE = "general.decision"
GENERAL_DECISION_MEMORY_SCHEMA_VERSION = "spice.memory.general.decision.v1"
GENERAL_REFLECTION_MEMORY_NAMESPACE = "general.reflection"
GENERAL_REFLECTION_MEMORY_SCHEMA_VERSION = "spice.memory.general.reflection.v1"


def write_general_decision_memory(
    provider: MemoryProvider,
    *,
    artifact: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a compact decision record for future decision context retrieval."""

    record = build_general_decision_memory_record(artifact=artifact)
    refs = _memory_refs(record)
    record_ids = provider.write(
        [record],
        namespace=GENERAL_DECISION_MEMORY_NAMESPACE,
        refs=refs,
    )
    session_summary = update_session_summary(provider, config=config)
    return {
        "enabled": True,
        "status": "written",
        "namespace": GENERAL_DECISION_MEMORY_NAMESPACE,
        "record_ids": record_ids,
        "refs": refs,
        "session_summary": session_summary,
    }


def skipped_general_decision_memory_writeback(*, reason: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "status": "skipped",
        "namespace": GENERAL_DECISION_MEMORY_NAMESPACE,
        "record_ids": [],
        "refs": [],
        "reason": reason,
    }


def write_general_reflection_memory(
    provider: MemoryProvider,
    *,
    decision_artifact: dict[str, Any],
    execution_artifact: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a compact execution outcome record for future reflection context."""

    record = build_general_reflection_memory_record(
        decision_artifact=decision_artifact,
        execution_artifact=execution_artifact,
    )
    refs = _memory_refs(record)
    record_ids = provider.write(
        [record],
        namespace=GENERAL_REFLECTION_MEMORY_NAMESPACE,
        refs=refs,
    )
    session_summary = update_session_summary(provider, config=config)
    return {
        "enabled": True,
        "status": "written",
        "namespace": GENERAL_REFLECTION_MEMORY_NAMESPACE,
        "record_ids": record_ids,
        "refs": refs,
        "session_summary": session_summary,
    }


def skipped_general_reflection_memory_writeback(*, reason: str) -> dict[str, Any]:
    return {
        "enabled": False,
        "status": "skipped",
        "namespace": GENERAL_REFLECTION_MEMORY_NAMESPACE,
        "record_ids": [],
        "refs": [],
        "reason": reason,
    }


def build_general_decision_memory_record(*, artifact: dict[str, Any]) -> dict[str, Any]:
    selected = _dict(artifact.get("compare_payload")).get("selected_recommendation")
    selected_payload = _dict(selected)
    decision_id = str(artifact.get("decision_id") or "")
    run_id = str(artifact.get("run_id") or "")
    state_after_ref = str(artifact.get("state_after_ref") or "")
    active_frame_ref = (
        f"{state_after_ref}#active_decision_frame:{decision_id}"
        if state_after_ref and decision_id
        else ""
    )
    return {
        "id": f"memory.general.decision.{run_id or decision_id}",
        "schema_version": GENERAL_DECISION_MEMORY_SCHEMA_VERSION,
        "record_type": "general.decision",
        "created_at": str(artifact.get("created_at") or ""),
        "session_id": str(artifact.get("session_id") or ""),
        "run_id": run_id,
        "decision_id": decision_id,
        "trace_ref": str(artifact.get("trace_ref") or ""),
        "source": str(artifact.get("source") or ""),
        "parent_run_id": str(artifact.get("parent_run_id") or ""),
        "run_intent_mode": str(artifact.get("run_intent_mode") or ""),
        "display_language": str(artifact.get("display_language") or ""),
        "input": payload_value(artifact.get("input") or {}),
        "candidate_summary": payload_value(artifact.get("candidate_summary") or {}),
        "selected": _compact_selected(selected_payload),
        "why_won": payload_value(selected_payload.get("decision_basis") or []),
        "approval_id": str(artifact.get("approval_id") or ""),
        "active_decision_frame_ref": active_frame_ref,
        "context_refs": payload_value(artifact.get("context_refs") or {}),
        "state_refs": {
            "before": str(artifact.get("state_before_ref") or ""),
            "after": state_after_ref,
        },
        "artifact_refs": payload_value(artifact.get("store_paths") or {}),
        "selection_pool": payload_value(artifact.get("selection_pool") or {}),
        "handoff": {
            "required": bool(artifact.get("handoff_required")),
            "blocked": bool(artifact.get("handoff_blocked")),
            "blockers": payload_value(artifact.get("handoff_blockers") or []),
            "approval_id": str(artifact.get("approval_id") or ""),
        },
        "active_decision_frame": _compact_active_decision_frame(
            _dict(artifact.get("active_decision_frame"))
        ),
    }


def build_general_reflection_memory_record(
    *,
    decision_artifact: dict[str, Any],
    execution_artifact: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(
        execution_artifact.get("selected_candidate_id")
        or execution_artifact.get("candidate_id")
        or ""
    )
    task_status = str(execution_artifact.get("task_status") or "")
    protocol_status = str(execution_artifact.get("protocol_status") or "")
    outcome_id = str(execution_artifact.get("outcome_id") or "")
    execution_id = str(execution_artifact.get("execution_id") or "")
    state_delta = _state_delta_summary(execution_artifact)
    return {
        "id": f"memory.general.reflection.{outcome_id or execution_id}",
        "schema_version": GENERAL_REFLECTION_MEMORY_SCHEMA_VERSION,
        "record_type": "general.reflection",
        "created_at": str(execution_artifact.get("created_at") or ""),
        "session_id": str(execution_artifact.get("session_id") or ""),
        "run_id": str(execution_artifact.get("run_id") or ""),
        "decision_id": str(execution_artifact.get("decision_id") or ""),
        "trace_ref": str(execution_artifact.get("trace_ref") or ""),
        "approval_id": str(execution_artifact.get("approval_id") or ""),
        "candidate_id": candidate_id,
        "selected_candidate": _compact_candidate_from_artifact(
            decision_artifact,
            candidate_id=candidate_id,
        ),
        "executor": {
            "provider": str(execution_artifact.get("executor_provider") or ""),
            "executor_id": str(execution_artifact.get("executor_id") or ""),
            "command": str(execution_artifact.get("executor_command") or ""),
            "skill_id": str(execution_artifact.get("skill_id") or ""),
            "context_pack_id": str(execution_artifact.get("context_pack_id") or ""),
            "dry_run": bool(execution_artifact.get("dry_run")),
            "executor_called": bool(execution_artifact.get("executor_called")),
            "real_executor_called": bool(execution_artifact.get("real_executor_called")),
            "sdep_request_sent": bool(execution_artifact.get("sdep_request_sent")),
            "executed": bool(execution_artifact.get("executed")),
        },
        "execution": {
            "execution_id": execution_id,
            "request_id": str(execution_artifact.get("request_id") or ""),
            "outcome_id": outcome_id,
            "protocol_status": protocol_status,
            "task_status": task_status,
            "success": _execution_succeeded(
                protocol_status=protocol_status,
                task_status=task_status,
            ),
            "state_updated": bool(execution_artifact.get("state_updated")),
            "state_before_ref": str(execution_artifact.get("state_before_ref") or ""),
            "state_after_ref": str(execution_artifact.get("state_after_ref") or ""),
        },
        "outcome_summary": _outcome_summary(execution_artifact),
        "state_delta_summary": state_delta,
        "artifact_refs": payload_value(execution_artifact.get("store_paths") or {}),
        "decision_memory_refs": _decision_memory_record_ids(decision_artifact),
    }


def _compact_selected(selected: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate_id": str(selected.get("candidate_id") or ""),
        "title": str(selected.get("title") or ""),
        "action": str(selected.get("action") or ""),
        "recommendation": str(
            selected.get("human_summary")
            or selected.get("recommendation")
            or selected.get("summary")
            or ""
        ),
        "score": selected.get("score"),
        "execution_affordance": payload_value(selected.get("execution_affordance") or {}),
        "skill_resolution": payload_value(selected.get("skill_resolution") or {}),
    }


def _compact_candidate_from_artifact(
    artifact: dict[str, Any],
    *,
    candidate_id: str,
) -> dict[str, Any]:
    if not candidate_id:
        return {}
    for candidate in _iter_candidate_payloads(artifact):
        if str(candidate.get("candidate_id") or candidate.get("id") or "") != candidate_id:
            continue
        return {
            "candidate_id": candidate_id,
            "title": str(candidate.get("title") or ""),
            "action": str(candidate.get("action") or candidate.get("action_type") or ""),
            "recommendation": str(
                candidate.get("human_summary")
                or candidate.get("recommendation")
                or candidate.get("summary")
                or ""
            ),
            "expected": str(
                candidate.get("expected_result")
                or candidate.get("expected")
                or ""
            ),
            "executor_task": str(candidate.get("executor_task") or ""),
            "execution_affordance": payload_value(
                candidate.get("execution_affordance") or {}
            ),
            "skill_resolution": payload_value(candidate.get("skill_resolution") or {}),
        }
    selected = _dict(_dict(artifact.get("compare_payload")).get("selected_recommendation"))
    if str(selected.get("candidate_id") or "") == candidate_id:
        return _compact_selected(selected)
    return {"candidate_id": candidate_id}


def _compact_active_decision_frame(frame: dict[str, Any]) -> dict[str, Any]:
    if not frame:
        return {}
    return {
        "decision_id": str(frame.get("decision_id") or ""),
        "run_id": str(frame.get("run_id") or ""),
        "status": str(frame.get("status") or ""),
        "selected_candidate_id": str(frame.get("selected_candidate_id") or ""),
        "approval_id": str(frame.get("approval_id") or ""),
        "allowed_continuations": payload_value(frame.get("allowed_continuations") or []),
        "selection_pool": payload_value(frame.get("selection_pool") or {}),
    }


def _memory_refs(record: dict[str, Any]) -> list[str]:
    refs = [
        str(record.get("run_id") or ""),
        str(record.get("decision_id") or ""),
        str(record.get("trace_ref") or ""),
        str(record.get("approval_id") or ""),
        str(record.get("candidate_id") or ""),
        str(record.get("active_decision_frame_ref") or ""),
    ]
    execution = record.get("execution")
    if isinstance(execution, dict):
        refs.extend(
            str(execution.get(key) or "")
            for key in ("execution_id", "request_id", "outcome_id", "state_after_ref")
        )
    context_refs = record.get("context_refs")
    if isinstance(context_refs, dict):
        refs.extend(str(value) for value in context_refs.values() if str(value or ""))
    return list(dict.fromkeys(ref for ref in refs if ref))


def _iter_candidate_payloads(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    compare = _dict(artifact.get("compare_payload"))
    for item in compare.get("candidate_decisions") or []:
        if isinstance(item, dict):
            payloads.append(item)
    for item in artifact.get("candidates") or []:
        if isinstance(item, dict):
            payloads.append(item)
    frame = _dict(artifact.get("active_decision_frame"))
    for item in frame.get("candidate_options") or []:
        if isinstance(item, dict):
            payloads.append(item)
    return payloads


def _execution_succeeded(*, protocol_status: str, task_status: str) -> bool:
    success_values = {"success", "succeeded", "ok", "completed"}
    protocol = protocol_status.strip().lower()
    task = task_status.strip().lower()
    if protocol and protocol not in success_values:
        return False
    return task in success_values


def _outcome_summary(artifact: dict[str, Any]) -> str:
    outcome_record = _dict(artifact.get("outcome_record"))
    summary = str(outcome_record.get("summary") or "")
    if summary:
        return summary
    output = _sdep_output(artifact)
    return str(output.get("summary") or "")


def _state_delta_summary(artifact: dict[str, Any]) -> dict[str, Any]:
    outcome_record = _dict(artifact.get("outcome_record"))
    state_delta = outcome_record.get("state_delta")
    if not isinstance(state_delta, dict):
        state_delta = _sdep_output(artifact).get("state_delta")
    if not isinstance(state_delta, dict):
        state_delta = {}
    return {
        "task_status": str(artifact.get("task_status") or ""),
        "state_updated": bool(artifact.get("state_updated")),
        "state_before_ref": str(artifact.get("state_before_ref") or ""),
        "state_after_ref": str(artifact.get("state_after_ref") or ""),
        "delta": payload_value(state_delta),
        "updated_refs": payload_value(state_delta.get("updated_refs") or []),
    }


def _sdep_output(artifact: dict[str, Any]) -> dict[str, Any]:
    response = _dict(artifact.get("sdep_response"))
    output = _dict(_dict(response.get("outcome")).get("output"))
    if output:
        return output
    outcome = _dict(artifact.get("outcome"))
    return _dict(_dict(_dict(outcome.get("sdep_response")).get("outcome")).get("output"))


def _decision_memory_record_ids(artifact: dict[str, Any]) -> list[str]:
    writeback = _dict(artifact.get("memory_writeback"))
    ids = writeback.get("record_ids")
    if not isinstance(ids, list):
        return []
    return [str(item) for item in ids if str(item or "")]


def _dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}
