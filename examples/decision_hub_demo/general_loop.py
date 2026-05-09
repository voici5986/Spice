from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from spice.decision.compare import render_compare_text

from examples.decision_hub_demo.general_adapter import run_general_read_only_path
from examples.decision_hub_demo.general_approval import build_general_approval_artifact
from examples.decision_hub_demo.general_execution import (
    approve_general_approval,
    build_general_execution_artifact,
)
from examples.decision_hub_demo.general_outcome import (
    build_general_outcome_artifact,
    build_general_sdep_response_fixture,
)
from examples.decision_hub_demo.general_state_feedback import (
    build_general_state_feedback_artifact,
)
from examples.decision_hub_demo.ids import timestamp_segment


GENERAL_LOOP_ARTIFACT_BUILDER = "decision_hub_demo.general_loop"


def build_general_loop_artifact(
    *,
    now: datetime | None = None,
    use_bars: bool = False,
) -> dict[str, Any]:
    """Build the full read-only General decision loop artifact.

    The loop is intentionally artifact-only. It plans the SDEP handoff and
    applies a fixture outcome observation to a state snapshot, but never calls
    an executor, sends SDEP, persists state, or mutates the legacy demo runtime.
    """

    created = now or datetime.now(timezone.utc).replace(microsecond=0)
    result = run_general_read_only_path(now=created)
    compare_payload = result.policy_result.compare_payload
    decision_card_text = render_compare_text(compare_payload, use_bars=use_bars)

    approval_artifact = build_general_approval_artifact(result, now=created)
    approval_payload = approval_artifact.get("approval")
    approved = (
        approve_general_approval(approval_payload, now=created)
        if isinstance(approval_payload, dict)
        else None
    )
    execution_artifact = build_general_execution_artifact(
        result,
        approval=approved,
        now=created,
    )
    response_payload = build_general_sdep_response_fixture(
        execution_artifact,
        now=created,
    )
    outcome_artifact = build_general_outcome_artifact(
        execution_artifact,
        response_payload,
        now=created,
    )
    state_feedback_artifact = build_general_state_feedback_artifact(
        result.state,
        outcome_artifact,
        now=created,
    )

    selected = compare_payload["selected_recommendation"]
    request_id = outcome_artifact.get("request_id") or _nested(
        execution_artifact,
        "sdep_request",
        "request_id",
    )
    observations = [observation.to_payload() for observation in result.observations]
    candidates = [candidate.to_payload() for candidate in result.candidates]
    artifact = {
        "path_type": "read_only_general_full_loop",
        "generated_by": GENERAL_LOOP_ARTIFACT_BUILDER,
        "created_at": timestamp_segment(created),
        "status": "full_loop_rendered",
        "loop_status": "completed_read_only",
        "decision_id": compare_payload["decision_id"],
        "trace_ref": compare_payload["trace_ref"],
        "selected_candidate_id": selected["candidate_id"],
        "approval_id": approval_artifact.get("approval", {}).get("approval_id")
        if isinstance(approval_artifact.get("approval"), dict)
        else None,
        "skill_id": execution_artifact.get("skill_id"),
        "executor_id": execution_artifact.get("executor_id"),
        "context_pack_id": execution_artifact.get("context_pack_id"),
        "execution_id": execution_artifact.get("execution_id"),
        "request_id": request_id,
        "outcome_id": outcome_artifact.get("outcome_id"),
        "protocol_status": outcome_artifact.get("protocol_status"),
        "task_status": outcome_artifact.get("task_status"),
        "read_only": True,
        "executor_called": False,
        "executed": False,
        "execution": None,
        "sdep_request_sent": False,
        "persisted": False,
        "state_snapshot_updated": True,
        "update_mode": "read_only_snapshot",
        "state_before_summary": state_feedback_artifact["state_before_summary"],
        "state_after_summary": state_feedback_artifact["state_after_summary"],
        "state_before": state_feedback_artifact["state_before"],
        "state_after": state_feedback_artifact["state_after"],
        "observations": observations,
        "candidates": candidates,
        "candidate_summary": _candidate_summary(
            candidates,
            selected_candidate_id=selected["candidate_id"],
        ),
        "resolved_skill": execution_artifact.get("resolved_skill"),
        "context_pack": execution_artifact.get("context_pack"),
        "compare_payload": compare_payload,
        "approval_artifact": approval_artifact,
        "execution_artifact": execution_artifact,
        "outcome_artifact": outcome_artifact,
        "state_feedback_artifact": state_feedback_artifact,
        "flow": [
            "observations",
            "general_state",
            "generic_candidates",
            "policy_decision",
            "approval_checkpoint",
            "skill_resolution",
            "context_pack",
            "sdep_request_plan",
            "sdep_response_fixture",
            "outcome_observation",
            "state_feedback_snapshot",
        ],
        "decision": {
            "observations": observations,
            "state_summary": _state_summary(result.state),
            "candidate_count": len(result.candidates),
            "candidates": candidates,
            "compare_payload": compare_payload,
            "rendered_text": decision_card_text,
        },
        "approval": approval_artifact,
        "execution_plan": execution_artifact,
        "outcome_return": outcome_artifact,
        "state_feedback": state_feedback_artifact,
    }
    artifact["rendered_text"] = render_general_loop_text(artifact)
    return artifact


def render_general_loop_text(artifact: dict[str, Any]) -> str:
    decision = _dict(artifact.get("decision"))
    compare_payload = _dict(decision.get("compare_payload"))
    selected = _dict(compare_payload.get("selected_recommendation"))
    state_summary = _dict(compare_payload.get("decision_relevant_state_summary"))
    approval = _dict(artifact.get("approval"))
    execution = _dict(artifact.get("execution_plan"))
    resolved_skill = _dict(artifact.get("resolved_skill"))
    context_pack = _dict(artifact.get("context_pack"))
    before = _dict(artifact.get("state_before_summary"))
    after = _dict(artifact.get("state_after_summary"))
    candidate_summary = _dict(artifact.get("candidate_summary"))
    approval_status = (
        _nested(approval, "approval_bridge", "status")
        or _nested(approval, "approval", "status")
        or approval.get("status")
    )

    lines = [
        "SPICE DECISION LOOP",
        "read-only preview: decision -> approval -> skill handoff -> outcome snapshot",
        "no executor called | no SDEP sent | no state persisted",
        "",
        f"decision_id: {artifact.get('decision_id')}",
        f"trace_ref: {artifact.get('trace_ref')}",
        f"selected_candidate_id: {artifact.get('selected_candidate_id')}",
        "",
        "0. INPUT SIGNALS",
        f"- normalized observations: {len(artifact.get('observations', []))}",
        f"- provider path: {artifact.get('generated_by')}",
        "",
        "1. GENERAL STATE",
        f"- commitments: {before.get('commitment_count')} active in state",
        f"- work items: {before.get('work_item_count')} open or active in state",
        f"- capabilities: {before.get('capability_count')} available in state",
        f"- decision-relevant: {_state_focus_line(state_summary)}",
        "",
        "2. CANDIDATE DECISIONS",
        f"- generated: {candidate_summary.get('total', 0)}",
        f"- selected candidate is included: {str(bool(candidate_summary.get('selected_present'))).lower()}",
    ]
    for candidate in _candidate_preview(compare_payload)[:5]:
        marker = "*" if candidate.get("is_selected") else "-"
        title = candidate.get("title") or candidate.get("candidate_id")
        action = candidate.get("action")
        lines.append(f"{marker} {title} [{action}]")

    lines.extend(
        [
            "",
            "3. SELECTED DECISION",
            f"- selected: {selected.get('title')} ({selected.get('candidate_id')})",
            f"- action: {selected.get('action')}",
        ]
    )
    for reason in _selection_basis(selected)[:3]:
        lines.append(f"- why: {reason}")

    lines.extend(["", "4. WHY NOT OTHERS"])
    why_not = _why_not_preview(compare_payload)[:2]
    if why_not:
        for reason in why_not:
            lines.append(f"- {reason}")
    else:
        lines.append("- no non-selected comparison reasons were recorded")

    lines.extend(
        [
            "",
            "5. APPROVAL CHECKPOINT",
            f"- status: {approval_status}",
            f"- approval_id: {artifact.get('approval_id')}",
            f"- execution_allowed: {str(bool(approval.get('execution_allowed'))).lower()}",
            f"- local approval fixture used for preview: {str(bool(execution.get('approved'))).lower()}",
            "",
            "6. EXECUTION HANDOFF",
            f"- skill: {artifact.get('skill_id')}",
            f"- skill_source: {_nested(resolved_skill, 'metadata', 'skill_source')}",
            f"- planned_executor: {artifact.get('executor_id')}",
            f"- context_pack_id: {artifact.get('context_pack_id')}",
            f"- context: compact pack, {len(context_pack.get('target_refs', []))} target ref(s)",
            f"- skill side effect: {resolved_skill.get('side_effect_class')}",
            f"- task: {context_pack.get('task')}",
            f"- why_now: {context_pack.get('why_now')}",
            f"- expected_output: {context_pack.get('expected_output')}",
            f"- sdep_request_sent: {str(bool(artifact.get('sdep_request_sent'))).lower()}",
            "",
            "7. EXECUTION BOUNDARY",
            f"- status: {execution.get('execution_status')} (planned only)",
            f"- protocol message: {_nested(execution, 'sdep_request', 'message_type')} (not sent)",
            f"- execution_id: {artifact.get('execution_id')}",
            f"- read_only: {str(bool(artifact.get('read_only'))).lower()}",
            f"- sdep_request_sent: {str(bool(artifact.get('sdep_request_sent'))).lower()}",
            f"- executor_called: {str(bool(artifact.get('executor_called'))).lower()}",
            f"- executed: {str(bool(artifact.get('executed'))).lower()}",
            f"- persisted: {str(bool(artifact.get('persisted'))).lower()}",
            "",
            "8. OUTCOME RETURN",
            f"- outcome_id: {artifact.get('outcome_id')}",
            f"- protocol_status: {artifact.get('protocol_status')}",
            f"- task_status: {artifact.get('task_status')}",
            "- source: local fixture response",
            "",
            "9. STATE FEEDBACK",
            f"- observations: {before.get('observation_count')} -> {after.get('observation_count')}",
            f"- outcomes: {before.get('outcome_count')} -> {after.get('outcome_count')}",
            f"- state_snapshot_updated: {str(bool(artifact.get('state_snapshot_updated'))).lower()}",
            f"- update_mode: {artifact.get('update_mode')}",
            f"- persisted: {str(bool(artifact.get('persisted'))).lower()}",
            "",
            "10. TRACE",
            f"- decision_id: {artifact.get('decision_id')}",
            f"- trace_ref: {artifact.get('trace_ref')}",
            f"- approval_id: {artifact.get('approval_id')}",
            f"- skill_id: {artifact.get('skill_id')}",
            f"- context_pack_id: {artifact.get('context_pack_id')}",
            f"- execution_id: {artifact.get('execution_id')}",
            f"- request_id: {artifact.get('request_id')}",
            f"- outcome_id: {artifact.get('outcome_id')}",
        ]
    )
    return "\n".join(lines)


def _state_summary(state: Any) -> dict[str, Any]:
    return {
        "state_id": state.state_id,
        "observation_count": len(state.observations),
        "intent_count": len(state.intents),
        "commitment_count": len(state.commitments),
        "work_item_count": len(state.work_items),
        "capability_count": len(state.capabilities),
        "constraint_count": len(state.constraints),
        "risk_count": len(state.risks),
        "open_loop_count": len(state.open_loops),
        "outcome_count": len(state.outcomes),
    }


def _candidate_summary(
    candidates: list[dict[str, Any]],
    *,
    selected_candidate_id: str,
) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    by_action_type: dict[str, int] = {}
    for candidate in candidates:
        status = str(candidate.get("availability_status") or "unknown")
        action_type = str(candidate.get("action_type") or "unknown")
        by_status[status] = by_status.get(status, 0) + 1
        by_action_type[action_type] = by_action_type.get(action_type, 0) + 1
    return {
        "total": len(candidates),
        "selected_candidate_id": selected_candidate_id,
        "selected_present": any(
            candidate.get("candidate_id") == selected_candidate_id
            for candidate in candidates
        ),
        "by_availability_status": by_status,
        "by_action_type": by_action_type,
    }


def _selection_basis(selected: dict[str, Any]) -> list[str]:
    rendered: list[str] = []
    for basis in selected.get("decision_basis", []):
        if not isinstance(basis, dict):
            continue
        kind = str(basis.get("kind", ""))
        if kind == "weighted_dimension":
            rendered.append(
                f"{basis.get('label')} carried weight {float(basis.get('weight', 0.0)):.2f}"
            )
        elif kind == "tradeoff_rule":
            rendered.append(f"trade-off rule {basis.get('rule_id')} supported it")
        elif basis.get("summary"):
            rendered.append(str(basis["summary"]))
    if not rendered and selected.get("selection_reason"):
        rendered.append(str(selected["selection_reason"]))
    return rendered


def _candidate_preview(compare_payload: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = compare_payload.get("candidate_decisions", [])
    if not isinstance(candidates, list):
        return []
    selected = [
        item for item in candidates
        if isinstance(item, dict) and item.get("is_selected")
    ]
    others = [
        item for item in candidates
        if isinstance(item, dict) and not item.get("is_selected")
    ]
    return selected + others


def _why_not_preview(compare_payload: dict[str, Any]) -> list[str]:
    rendered: list[str] = []
    entries = compare_payload.get("why_not_the_others", [])
    if not isinstance(entries, list):
        return rendered
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        reason = _first_reason(entry.get("reasons"))
        if reason:
            title = entry.get("title") or entry.get("candidate_id") or "candidate"
            rendered.append(f"{title}: {reason}")
    return rendered


def _first_reason(value: Any) -> str:
    if not isinstance(value, list):
        return ""
    for item in value:
        if not isinstance(item, dict):
            continue
        kind = item.get("kind")
        if kind == "weighted_dimension_gap":
            label = item.get("label") or item.get("dimension")
            weight = float(item.get("weight") or 0.0)
            return f"selected candidate had stronger {label} contribution at weight {weight:.2f}"
        if kind == "veto":
            return f"blocked by {item.get('constraint_id') or item.get('reason')}"
        if item.get("summary"):
            return str(item["summary"])
    return ""


def _state_focus_line(summary: dict[str, Any]) -> str:
    commitments = summary.get("active_commitments") or []
    work_items = summary.get("open_work_items") or []
    conflicts = summary.get("active_conflicts") or []
    parts: list[str] = []
    if isinstance(commitments, list) and commitments:
        parts.append(f"{len(commitments)} commitment")
    if isinstance(work_items, list) and work_items:
        parts.append(f"{len(work_items)} work item")
    if isinstance(conflicts, list) and conflicts:
        parts.append(f"{len(conflicts)} conflict signal")
    if summary.get("executor_available") is True:
        parts.append("executor available")
    return ", ".join(parts) if parts else "no focused state summary recorded"


def _dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _nested(payload: dict[str, Any], *keys: str) -> Any:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
