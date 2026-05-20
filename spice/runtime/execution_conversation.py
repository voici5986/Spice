from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from spice.decision.general.approval import Approval
from spice.language import detect_display_language
from spice.memory import MemoryProvider
from spice.runtime.conversation import build_conversation_turn, save_conversation_turn
from spice.runtime.memory_writeback import (
    skipped_general_evolution_memory_writeback,
    write_general_evolution_memory,
)
from spice.runtime.session import DEFAULT_SESSION_ID, SessionRecord, load_or_create_session
from spice.runtime.store import LocalJsonStore


EXECUTION_CONVERSATION_SCHEMA_VERSION = "spice.execution_conversation.v1"


@dataclass(frozen=True, slots=True)
class ExecutionConversationResult:
    approval_id: str
    rendered_text: str
    conversation_turn_id: str
    response_id: str
    candidate_id: str
    decision_id: str
    run_id: str
    artifact: dict[str, Any]


def open_execution_approval_from_frame(
    *,
    store: LocalJsonStore,
    session_id: str = DEFAULT_SESSION_ID,
    user_input: str,
    active_frame: Mapping[str, Any],
    candidate_id: str = "",
    memory_provider: MemoryProvider | None = None,
    now: datetime | None = None,
) -> ExecutionConversationResult:
    created = now or datetime.now(timezone.utc)
    frame = _frame_with_selected_candidate(active_frame, candidate_id)
    selected = _mapping(frame.get("selected"))
    selected_id = str(selected.get("candidate_id") or frame.get("selected_candidate_id") or "").strip()
    if not selected_id:
        raise ValueError("Active Decision Card has no selected candidate to execute.")
    eligibility = _execution_approval_eligibility(selected)
    if not eligibility["eligible"]:
        raise ValueError(
            _execution_approval_block_message(
                selected=selected,
                reason=str(eligibility["reason"] or ""),
                language=detect_display_language(user_input),
            )
        )

    existing_approval_id = str(frame.get("approval_id") or "").strip()
    approval = (
        _load_existing_approval(store, existing_approval_id)
        if existing_approval_id
        else None
    )
    if approval is None:
        approval = _build_pending_approval(
            frame=frame,
            selected=selected,
            candidate_id=selected_id,
            created=created,
        )
        store.save_approval(approval.approval_id, approval.to_payload())

    updated_frame = _attach_approval_to_frame(
        frame,
        approval_id=approval.approval_id,
        updated_at=_timestamp(created),
    )
    _save_active_frame_to_state(store, updated_frame, approval.to_payload())
    _sync_source_artifacts(store, updated_frame, approval.to_payload())

    session = _append_approval_to_session(
        store,
        load_or_create_session(store, session_id=session_id, now=created),
        approval_id=approval.approval_id,
        now=created,
    )
    turn = build_conversation_turn(
        user_input=user_input,
        route="execution_request",
        session_id=session.session_id,
        created_at=created,
        source_decision_id=str(updated_frame.get("decision_id") or ""),
        source_candidate_id=selected_id,
        source_run_id=str(updated_frame.get("run_id") or ""),
        source_approval_id=approval.approval_id,
        artifact_refs=_artifact_refs(store, updated_frame, approval.approval_id),
        metadata={
            "execution_conversation_action": "open_pending_approval",
            "candidate_label": str(selected.get("label") or ""),
            "schema_version": EXECUTION_CONVERSATION_SCHEMA_VERSION,
        },
    )
    response = _response_artifact(
        turn_id=turn.turn_id,
        response_id=turn.response_id or "",
        user_input=user_input,
        frame=updated_frame,
        selected=selected,
        approval=approval,
        created_at=_timestamp(created),
    )
    turn.metadata["execution_conversation_response"] = response
    saved_turn_path = save_conversation_turn(store, turn)
    _append_conversation_to_session(store, session.session_id, turn.turn_id, now=created)

    rendered = _render_execution_conversation(
        selected=selected,
        approval=approval,
        language=detect_display_language(user_input),
        already_pending=bool(existing_approval_id),
    )
    response["rendered_text"] = rendered
    response["evolution_memory_writeback"] = _write_execution_request_evolution_memory(
        memory_provider,
        record={
            "created_at": _timestamp(created),
            "session_id": session.session_id,
            "turn_id": turn.turn_id,
            "response_id": turn.response_id,
            "user_input": user_input,
            "route": "execution_request",
            "route_result": {
                "route": "execution_request",
                "action": "open_pending_approval",
                "candidate_id": selected_id,
                "decision_id": str(updated_frame.get("decision_id") or ""),
                "run_id": str(updated_frame.get("run_id") or ""),
                "approval_id": approval.approval_id,
            },
            "response_summary": _response_summary(rendered),
            "decision_id": str(updated_frame.get("decision_id") or ""),
            "run_id": str(updated_frame.get("run_id") or ""),
            "trace_ref": str(updated_frame.get("trace_ref") or ""),
            "candidate_id": selected_id,
            "selected_candidate": dict(selected),
            "follow_up_type": "execution_request",
            "approval_id": approval.approval_id,
            "approval": approval.to_payload(),
            "artifact_refs": turn.artifact_refs,
            "conversation_turn": turn.to_payload(),
            "metadata": {
                "generated_by": "spice.runtime.execution_conversation",
                "source": "execution_conversation_response",
            },
        },
    )
    turn.metadata["execution_conversation_response"] = response
    store.save_conversation_turn(turn.turn_id, {**turn.to_payload(), "metadata": turn.metadata})

    return ExecutionConversationResult(
        approval_id=approval.approval_id,
        rendered_text=rendered,
        conversation_turn_id=turn.turn_id,
        response_id=turn.response_id or "",
        candidate_id=selected_id,
        decision_id=str(updated_frame.get("decision_id") or ""),
        run_id=str(updated_frame.get("run_id") or ""),
        artifact={**response, "conversation_turn_path": _workspace_relative(saved_turn_path)},
    )


def _write_execution_request_evolution_memory(
    memory_provider: MemoryProvider | None,
    *,
    record: dict[str, Any],
) -> dict[str, Any]:
    if memory_provider is None:
        return skipped_general_evolution_memory_writeback(reason="memory_provider_not_configured")
    try:
        return write_general_evolution_memory(memory_provider, record=record)
    except Exception as exc:
        return skipped_general_evolution_memory_writeback(reason=f"write_failed:{exc}")


def _response_summary(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:240]
    return ""


def _build_pending_approval(
    *,
    frame: Mapping[str, Any],
    selected: Mapping[str, Any],
    candidate_id: str,
    created: datetime,
) -> Approval:
    affordance = _mapping(selected.get("execution_affordance"))
    permission = _mapping(affordance.get("permission"))
    executor = _mapping(affordance.get("executor"))
    required_permission = str(permission.get("required") or "workspace_write")
    title = _candidate_title(selected)
    executor_id = str(executor.get("executor_id") or "configured executor")
    approval_id = _approval_id_for_frame(frame, candidate_id)
    return Approval(
        approval_id=approval_id,
        decision_id=str(frame.get("decision_id") or ""),
        candidate_id=candidate_id,
        status="pending",
        mode="confirm_before_execution",
        requested_at=_timestamp(created),
        execution_allowed=False,
        prompt=(
            f"Approve executor handoff for {title}? "
            f"Executor={executor_id}; permission={required_permission}."
        ),
        metadata={
            "created_by": "spice.runtime.execution_conversation",
            "required_executor_permission": required_permission,
            "required_executor_permission_reason": str(permission.get("reason") or ""),
            "executor_available": bool(affordance.get("executor_available")),
            "executor_id": executor_id,
            "source_run_id": str(frame.get("run_id") or ""),
            "source_frame_id": str(frame.get("frame_id") or ""),
            "candidate_label": str(selected.get("label") or ""),
            "permission_requirement": {
                "required_permission": required_permission,
                "reason": str(permission.get("reason") or ""),
                "source": str(permission.get("source") or "execution_affordance"),
                "side_effect_class": str(permission.get("side_effect_class") or ""),
            },
            "execution_affordance": affordance,
        },
    )


def _execution_approval_eligibility(selected: Mapping[str, Any]) -> dict[str, Any]:
    affordance = _mapping(selected.get("execution_affordance"))
    if not affordance:
        return {
            "eligible": False,
            "reason": "selected candidate has no runtime execution affordance",
        }
    approval = _mapping(affordance.get("approval"))
    hard_block_reason = _candidate_hard_block_reason(selected, affordance)
    if hard_block_reason:
        return {
            "eligible": False,
            "reason": hard_block_reason,
        }
    if "candidate_execution_requested" in affordance and not bool(affordance.get("candidate_execution_requested")):
        return {
            "eligible": False,
            "reason": "selected candidate is advisory-only; no executor handoff was requested",
        }
    if not bool(affordance.get("candidate_executable")):
        return {
            "eligible": False,
            "reason": str(affordance.get("blocked_reason") or "selected candidate is not executable"),
        }
    if not bool(affordance.get("executor_available")):
        executor = _mapping(affordance.get("executor"))
        return {
            "eligible": False,
            "reason": str(executor.get("detail") or "configured executor is not available"),
        }
    if not bool(affordance.get("executable")):
        return {
            "eligible": False,
            "reason": str(affordance.get("blocked_reason") or "selected candidate cannot cross the execution boundary"),
        }
    if not bool(approval.get("required")):
        return {
            "eligible": False,
            "reason": "selected candidate does not require execution approval",
        }
    if not bool(approval.get("eligible_for_approval")):
        return {
            "eligible": False,
            "reason": str(approval.get("reason") or "selected candidate is not eligible for approval"),
        }
    return {"eligible": True, "reason": ""}


def _candidate_hard_block_reason(
    selected: Mapping[str, Any],
    affordance: Mapping[str, Any],
) -> str:
    text = _candidate_boundary_text(selected)
    if _is_noop_or_defer_candidate(text):
        return (
            "selected candidate is no-op/defer/record-only; it records or postpones the decision "
            "and cannot be sent to an executor"
        )
    if _is_read_only_candidate(selected, affordance, text):
        return (
            "selected candidate is read-only; perception, inspection, and advisory reads "
            "do not cross the execution approval boundary"
        )
    return ""


def _candidate_boundary_text(selected: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for key in (
        "action",
        "intent",
        "title",
        "recommended_action",
        "expected_result",
        "executor_task",
        "required_capability",
    ):
        value = selected.get(key)
        if value:
            parts.append(str(value))
    for item in _list(selected.get("why_now")):
        if item:
            parts.append(str(item))
    return "\n".join(parts).lower()


def _is_noop_or_defer_candidate(text: str) -> bool:
    patterns = (
        "time.defer",
        "state.record",
        "record-only",
        "record only",
        "no-op",
        "noop",
        "defer",
        "later",
        "skip",
        "postpone",
        "暂缓",
        "稍后",
        "仅记录",
        "只记录",
        "记录状态",
        "不执行",
        "不要现在发起",
        "先不要现在发起",
    )
    return any(pattern in text for pattern in patterns)


def _is_read_only_candidate(
    selected: Mapping[str, Any],
    affordance: Mapping[str, Any],
    text: str,
) -> bool:
    read_only_terms = (
        "read-only",
        "read_only",
        "read_file",
        "repo_map",
        "search",
        "git_status",
        "git_diff",
        "git_log",
        "read_package_metadata",
        "read_test_structure",
        "read_python_symbol",
        "workspace perception",
        "inspect current implementation",
        "读取",
        "查看当前实现",
        "读 repo",
    )
    state_changing_terms = (
        "write_file",
        "patch",
        "edit",
        "delete",
        "move",
        "install",
        "terminal_command",
        "run test",
        "pytest",
        "修改",
        "写入",
        "删除",
        "安装",
        "执行测试",
    )
    return any(term in text for term in read_only_terms) and not any(term in text for term in state_changing_terms)


def _execution_approval_block_message(
    *,
    selected: Mapping[str, Any],
    reason: str,
    language: str,
) -> str:
    title = _candidate_title(selected)
    if language == "zh":
        return "\n".join(
            [
                f"当前选择不是可执行任务，不能生成 execution approval：{title}",
                f"原因：{reason or 'selected candidate is not executable'}",
                "Next: refine 当前决策，选择一个可执行候选，或使用 `/act <具体可执行任务>` 创建新的 approval-gated handoff。",
            ]
        )
    return "\n".join(
        [
            f"Selected candidate is not executable, so I cannot create execution approval: {title}",
            f"Reason: {reason or 'selected candidate is not executable'}",
            "Next: refine the decision, choose an executable candidate, or use `/act <specific executable task>` to create a new approval-gated handoff.",
        ]
    )


def _render_execution_conversation(
    *,
    selected: Mapping[str, Any],
    approval: Approval,
    language: str,
    already_pending: bool,
) -> str:
    affordance = _mapping(selected.get("execution_affordance"))
    executor = _mapping(affordance.get("executor"))
    permission = _mapping(affordance.get("permission"))
    executor_id = str(executor.get("executor_id") or approval.metadata.get("executor_id") or "configured executor")
    required_permission = str(
        permission.get("required")
        or approval.metadata.get("required_executor_permission")
        or "workspace_write"
    )
    title = _candidate_title(selected)
    if language == "zh":
        prefix = "已有一个 pending approval。" if already_pending else "我已经生成了一个 pending approval。"
        return "\n".join(
            [
                f"将按当前选择执行：{title}",
                f"这会交给 {executor_id} 执行，并需要 {required_permission} 权限。",
                prefix,
                f"approval_id: {approval.approval_id}",
                "",
                "Next:",
                f"- approve and execute: `y` 或 `/execute {approval.approval_id}`",
                f"- approve only: `a` 或 `/approve {approval.approval_id}`",
                f"- inspect: `/details {approval.approval_id}`",
            ]
        )
    prefix = "There is already a pending approval." if already_pending else "I generated a pending approval."
    return "\n".join(
        [
            f"I'll use the current selection: {title}",
            f"This will hand off to {executor_id} and requires {required_permission} permission.",
            prefix,
            f"approval_id: {approval.approval_id}",
            "",
            "Next:",
            f"- approve and execute: `y` or `/execute {approval.approval_id}`",
            f"- approve only: `a` or `/approve {approval.approval_id}`",
            f"- inspect: `/details {approval.approval_id}`",
        ]
    )


def _response_artifact(
    *,
    turn_id: str,
    response_id: str,
    user_input: str,
    frame: Mapping[str, Any],
    selected: Mapping[str, Any],
    approval: Approval,
    created_at: str,
) -> dict[str, Any]:
    return {
        "schema_version": EXECUTION_CONVERSATION_SCHEMA_VERSION,
        "created_at": created_at,
        "turn_id": turn_id,
        "response_id": response_id,
        "route": "execution_request",
        "action": "open_pending_approval",
        "user_input": user_input,
        "decision_id": str(frame.get("decision_id") or ""),
        "run_id": str(frame.get("run_id") or ""),
        "candidate_id": approval.candidate_id,
        "candidate_label": str(selected.get("label") or ""),
        "candidate_title": _candidate_title(selected),
        "approval_id": approval.approval_id,
        "approval": approval.to_payload(),
    }


def _frame_with_selected_candidate(
    active_frame: Mapping[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    frame = dict(active_frame)
    target_id = str(candidate_id or "").strip()
    if not target_id:
        return frame
    candidates = [_mapping(candidate) for candidate in _list(frame.get("candidates"))]
    selected: dict[str, Any] | None = None
    for candidate in candidates:
        is_selected = str(candidate.get("candidate_id") or "") == target_id
        candidate["is_selected"] = is_selected
        if is_selected:
            selected = dict(candidate)
    if selected is None:
        return frame
    frame["candidates"] = candidates
    frame["selected"] = selected
    frame["selected_candidate_id"] = target_id
    if str(frame.get("approval_id") or "") and target_id != str(active_frame.get("selected_candidate_id") or ""):
        frame["approval_id"] = ""
    return frame


def _attach_approval_to_frame(
    frame: Mapping[str, Any],
    *,
    approval_id: str,
    updated_at: str,
) -> dict[str, Any]:
    updated = dict(frame)
    updated["approval_id"] = approval_id
    updated["status"] = "approval_pending"
    updated["updated_at"] = updated_at
    continuations = [
        item for item in _list(updated.get("allowed_continuations"))
        if _mapping(item).get("action") not in {"approve_execute", "approve_only", "reject"}
    ]
    updated["allowed_continuations"] = [
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
        *continuations,
    ]
    return updated


def _save_active_frame_to_state(
    store: LocalJsonStore,
    frame: Mapping[str, Any],
    approval: Mapping[str, Any],
) -> None:
    payload = store.load_state()
    general = _general_state_payload(payload)
    metadata = _mapping(general.get("metadata"))
    metadata["active_decision_frame"] = dict(frame)
    general["metadata"] = metadata
    approvals = _list(general.get("approvals"))
    approval_id = str(approval.get("approval_id") or "")
    if approval_id and not any(_mapping(item).get("approval_id") == approval_id for item in approvals):
        approvals.append(dict(approval))
    general["approvals"] = approvals
    store.save_state(payload)


def _sync_source_artifacts(
    store: LocalJsonStore,
    frame: Mapping[str, Any],
    approval: Mapping[str, Any],
) -> None:
    approval_id = str(approval.get("approval_id") or "")
    run_id = str(frame.get("run_id") or "")
    if run_id:
        try:
            run = store.load_run(run_id)
        except FileNotFoundError:
            run = {}
        if run:
            run["approval_id"] = approval_id
            run["approval"] = dict(approval)
            run["active_decision_frame"] = dict(frame)
            run.setdefault("store_paths", {})["approval"] = _workspace_relative(
                store.record_path("approval", approval_id)
            )
            decision = _mapping(run.get("decision"))
            if decision:
                _replace_checkpoint_approval(decision, approval)
                run["decision"] = decision
            store.save_run(run_id, run)

    decision_id = str(frame.get("decision_id") or "")
    if decision_id:
        try:
            decision = store.load_decision(decision_id)
        except FileNotFoundError:
            decision = {}
        if decision:
            _replace_checkpoint_approval(decision, approval)
            store.save_decision(decision_id, decision)


def _append_approval_to_session(
    store: LocalJsonStore,
    session: SessionRecord,
    *,
    approval_id: str,
    now: datetime,
) -> SessionRecord:
    updated_at = _timestamp(now)
    updated = SessionRecord(
        session_id=session.session_id,
        created_at=session.created_at or updated_at,
        updated_at=updated_at,
        status=session.status,
        run_ids=list(session.run_ids),
        decision_ids=list(session.decision_ids),
        approval_ids=_append_unique(session.approval_ids, approval_id),
        conversation_turn_ids=list(session.conversation_turn_ids),
        active_state_ref=session.active_state_ref,
        last_run_id=session.last_run_id,
        last_decision_id=session.last_decision_id,
        last_trace_ref=session.last_trace_ref,
        pending_approval_ids=_append_unique(session.pending_approval_ids, approval_id),
        metadata=dict(session.metadata),
    )
    store.save_session(updated.session_id, updated.to_payload())
    return updated


def _append_conversation_to_session(
    store: LocalJsonStore,
    session_id: str,
    turn_id: str,
    *,
    now: datetime,
) -> None:
    session = load_or_create_session(store, session_id=session_id, now=now)
    updated_at = _timestamp(now)
    updated = SessionRecord(
        session_id=session.session_id,
        created_at=session.created_at or updated_at,
        updated_at=updated_at,
        status=session.status,
        run_ids=list(session.run_ids),
        decision_ids=list(session.decision_ids),
        approval_ids=list(session.approval_ids),
        conversation_turn_ids=_append_unique(session.conversation_turn_ids, turn_id),
        active_state_ref=session.active_state_ref,
        last_run_id=session.last_run_id,
        last_decision_id=session.last_decision_id,
        last_trace_ref=session.last_trace_ref,
        pending_approval_ids=list(session.pending_approval_ids),
        metadata=dict(session.metadata),
    )
    store.save_session(updated.session_id, updated.to_payload())


def _replace_checkpoint_approval(payload: dict[str, Any], approval: Mapping[str, Any]) -> None:
    checkpoint = _mapping(payload.get("checkpoint"))
    if checkpoint:
        checkpoint["approval"] = dict(approval)
        payload["checkpoint"] = checkpoint


def _artifact_refs(
    store: LocalJsonStore,
    frame: Mapping[str, Any],
    approval_id: str,
) -> dict[str, str]:
    refs = {
        "approval": _workspace_relative(store.record_path("approval", approval_id)),
        "state": _workspace_relative(store.paths.state),
    }
    run_id = str(frame.get("run_id") or "")
    decision_id = str(frame.get("decision_id") or "")
    if run_id:
        refs["run"] = _workspace_relative(store.record_path("run", run_id))
    if decision_id:
        refs["decision"] = _workspace_relative(store.record_path("decision", decision_id))
    return refs


def _load_existing_approval(store: LocalJsonStore, approval_id: str) -> Approval | None:
    if not approval_id:
        return None
    try:
        payload = store.load_approval(approval_id)
    except FileNotFoundError:
        return None
    return Approval(
        approval_id=str(payload.get("approval_id") or approval_id),
        decision_id=str(payload.get("decision_id") or ""),
        candidate_id=str(payload.get("candidate_id") or ""),
        status=str(payload.get("status") or "pending"),
        mode=str(payload.get("mode") or "confirm_before_execution"),
        requested_at=str(payload.get("requested_at") or ""),
        resolved_at=str(payload.get("resolved_at") or ""),
        actor=str(payload.get("actor") or "user"),
        prompt=str(payload.get("prompt") or ""),
        response=str(payload.get("response") or ""),
        reason=str(payload.get("reason") or ""),
        execution_allowed=bool(payload.get("execution_allowed")),
        metadata=_mapping(payload.get("metadata")),
    )


def _approval_id_for_frame(frame: Mapping[str, Any], candidate_id: str) -> str:
    decision_id = str(frame.get("decision_id") or "decision")
    digest = sha256(f"{decision_id}\n{candidate_id}".encode("utf-8")).hexdigest()[:10]
    return f"approval.execution_request.{_safe_id(decision_id)}.{digest}"


def _candidate_title(candidate: Mapping[str, Any]) -> str:
    return (
        str(candidate.get("title") or "").strip()
        or str(candidate.get("recommended_action") or "").strip()
        or str(candidate.get("intent") or "").strip()
        or str(candidate.get("candidate_id") or "selected candidate")
    )


def _general_state_payload(payload: dict[str, Any]) -> dict[str, Any]:
    world = payload.setdefault("world_state", {})
    if not isinstance(world, dict):
        raise ValueError("state.world_state must be an object")
    domain = world.setdefault("domain_state", {})
    if not isinstance(domain, dict):
        raise ValueError("state.world_state.domain_state must be an object")
    general = domain.setdefault("general_decision", {})
    if not isinstance(general, dict):
        raise ValueError("state general_decision must be an object")
    return general


def _append_unique(items: list[str], value: str) -> list[str]:
    result = list(items)
    if value and value not in result:
        result.append(value)
    return result


def _safe_id(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in value)
    return safe.strip("._") or "id"


def _workspace_relative(path: Path) -> str:
    parts = path.parts
    if ".spice" in parts:
        index = parts.index(".spice")
        return str(Path(*parts[index:]))
    return str(path)


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []
