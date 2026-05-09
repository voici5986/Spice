from __future__ import annotations

from typing import Any, Mapping

from spice.decision.general.state import GeneralDecisionState


def build_active_decision_frame(
    *,
    compare_payload: Mapping[str, Any],
    run_id: str,
    session_id: str,
    input_text: str,
    created_at: str,
    run_intent_mode: str,
    display_language: str = "en",
    approval_id: str | None = None,
    selection_pool: Mapping[str, Any] | None = None,
    handoff_blocked: bool = False,
    handoff_blockers: list[str] | None = None,
    source: str = "manual_intent",
    parent_run_id: str = "",
) -> dict[str, Any]:
    candidates = _frame_candidates(compare_payload)
    selected = _selected_frame_candidate(compare_payload, candidates)
    return {
        "schema_version": "0.1",
        "frame_id": f"frame.{compare_payload.get('decision_id', run_id)}",
        "status": _frame_status(
            approval_id=approval_id,
            handoff_blocked=handoff_blocked,
            selected=selected,
        ),
        "created_at": created_at,
        "updated_at": created_at,
        "source": source,
        "run_id": run_id,
        "session_id": session_id,
        "parent_run_id": parent_run_id,
        "decision_id": str(compare_payload.get("decision_id") or ""),
        "trace_ref": str(compare_payload.get("trace_ref") or ""),
        "run_intent_mode": run_intent_mode,
        "display_language": display_language,
        "input": {
            "text": input_text,
        },
        "selected_candidate_id": selected.get("candidate_id", ""),
        "selected": selected,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "approval_id": approval_id or "",
        "handoff_blocked": bool(handoff_blocked),
        "handoff_blockers": list(handoff_blockers or []),
        "selection_pool": dict(selection_pool or {}),
        "allowed_continuations": _allowed_continuations(
            selected=selected,
            approval_id=approval_id,
            handoff_blocked=handoff_blocked,
        ),
    }


def attach_active_decision_frame(
    state: GeneralDecisionState,
    frame: Mapping[str, Any],
) -> None:
    metadata = dict(state.metadata or {})
    metadata["active_decision_frame"] = dict(frame)
    state.metadata = metadata


def _frame_candidates(compare_payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    score_candidates = _mapping(compare_payload.get("score_breakdown")).get("candidates", {})
    result: list[dict[str, Any]] = []
    for candidate in _list(compare_payload.get("candidate_decisions")):
        if not isinstance(candidate, Mapping):
            continue
        candidate_id = str(candidate.get("candidate_id") or "")
        score = _mapping(score_candidates).get(candidate_id, {})
        result.append(
            {
                "label": "",
                "candidate_id": candidate_id,
                "title": str(candidate.get("title") or ""),
                "action": str(candidate.get("action") or ""),
                "intent": str(candidate.get("intent") or ""),
                "recommended_action": str(candidate.get("recommended_action") or ""),
                "why_now": [str(item) for item in _list(candidate.get("why_now"))],
                "expected_result": str(candidate.get("expected_result") or ""),
                "executor_task": str(candidate.get("executor_task") or ""),
                "requires_confirmation": bool(candidate.get("requires_confirmation")),
                "is_selected": bool(candidate.get("is_selected")),
                "score_total": _number(_mapping(score).get("score_total")),
                "execution_affordance": _mapping(candidate.get("execution_affordance")),
                "skill_resolution": _mapping(candidate.get("skill_resolution")),
                "simulation": _mapping(candidate.get("simulation")),
            }
        )
    return _label_candidates_for_display(result)


def _selected_frame_candidate(
    compare_payload: Mapping[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    selected = _mapping(compare_payload.get("selected_recommendation"))
    selected_id = str(selected.get("candidate_id") or "")
    for candidate in candidates:
        if candidate.get("candidate_id") == selected_id:
            return dict(candidate)
    return {
        "label": "",
        "candidate_id": selected_id,
        "title": str(selected.get("title") or ""),
        "action": str(selected.get("action") or ""),
        "intent": "",
        "recommended_action": str(selected.get("human_summary") or ""),
        "why_now": [str(item) for item in _list(selected.get("reason_summary"))],
        "expected_result": "",
        "executor_task": "",
        "requires_confirmation": bool(selected.get("requires_confirmation")),
        "is_selected": True,
        "score_total": None,
        "execution_affordance": _mapping(selected.get("execution_affordance")),
        "skill_resolution": _mapping(selected.get("skill_resolution")),
        "simulation": _mapping(selected.get("simulation")),
    }


def _allowed_continuations(
    *,
    selected: Mapping[str, Any],
    approval_id: str | None,
    handoff_blocked: bool,
) -> list[dict[str, Any]]:
    affordance = _mapping(selected.get("execution_affordance"))
    approval = _mapping(affordance.get("approval"))
    executable = bool(affordance.get("executable"))
    continuations = [
        {
            "action": "choose_option",
            "aliases": ["A", "B", "C", "choose A", "go with B", "选A", "选第一个"],
            "description": "Choose one visible candidate from the current Decision Card.",
        },
        {
            "action": "refine",
            "aliases": ["refine", "make it safer", "换个方向", "调整一下"],
            "description": "Refine the current Decision Card with new feedback.",
        },
        {
            "action": "show_details",
            "aliases": ["details", "why", "展开", "详情"],
            "description": "Inspect the current Decision Card and approval context.",
        },
        {
            "action": "skip",
            "aliases": ["skip", "later", "先跳过"],
            "description": "Leave this Decision Card unchanged for now.",
        },
        {
            "action": "new_intent",
            "aliases": ["new", "start over", "新的问题"],
            "description": "Treat the next message as a new intent instead of a continuation.",
        },
    ]
    if approval_id:
        continuations[0:0] = [
            {
                "action": "approve_execute",
                "aliases": ["y", "yes", "approve and execute", "执行", "批准并执行"],
                "description": "Approve the selected candidate and execute with the configured executor.",
            },
            {
                "action": "approve_only",
                "aliases": ["a", "approve", "只批准"],
                "description": "Approve the selected candidate without executing yet.",
            },
            {
                "action": "reject",
                "aliases": ["n", "no", "reject", "拒绝"],
                "description": "Reject this approval request.",
            },
        ]
    elif executable or approval.get("eligible_for_approval"):
        continuations.insert(
            0,
            {
                "action": "act_on_selected",
                "aliases": ["act", "execute this", "make it happen", "执行这个"],
                "description": "Open an approval-gated execution path for the selected candidate.",
            },
        )
    if handoff_blocked:
        continuations.insert(
            0,
            {
                "action": "resolve_handoff_blocker",
                "aliases": ["fix blocker", "enable execution", "解决阻塞"],
                "description": "Resolve why the selected decision cannot cross the executor boundary.",
            },
        )
    return continuations


def _frame_status(
    *,
    approval_id: str | None,
    handoff_blocked: bool,
    selected: Mapping[str, Any],
) -> str:
    if approval_id:
        return "approval_pending"
    if handoff_blocked:
        return "handoff_blocked"
    affordance = _mapping(selected.get("execution_affordance"))
    if affordance.get("executable"):
        return "execution_ready"
    return "recommended"


def _candidate_label(index: int) -> str:
    if 0 <= index < 26:
        return chr(ord("A") + index)
    return str(index + 1)


def _label_candidates_for_display(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    visible = _visible_candidates(candidates, max_candidates=3)
    visible_ids = {str(candidate.get("candidate_id") or "") for candidate in visible}
    hidden = [
        candidate
        for candidate in candidates
        if str(candidate.get("candidate_id") or "") not in visible_ids
    ]
    ordered = [dict(candidate) for candidate in [*visible, *hidden]]
    for index, candidate in enumerate(ordered):
        candidate["label"] = _candidate_label(index)
    return ordered


def _visible_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_candidates: int,
) -> list[dict[str, Any]]:
    if max_candidates <= 0 or len(candidates) <= max_candidates:
        return list(candidates)
    selected = [candidate for candidate in candidates if candidate.get("is_selected")]
    non_selected = [candidate for candidate in candidates if not candidate.get("is_selected")]
    non_selected.sort(key=lambda candidate: float(candidate.get("score_total") or 0.0), reverse=True)
    visible: list[dict[str, Any]] = []
    if selected:
        visible.append(selected[0])
    for candidate in non_selected:
        if len(visible) >= max_candidates:
            break
        visible.append(candidate)
    return visible


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _number(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
