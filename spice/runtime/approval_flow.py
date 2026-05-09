from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from spice.decision.general.approval import Approval
from spice.decision.general.types import safe_dataclass_from_payload
from spice.runtime.session import SessionRecord
from spice.runtime.store import LocalJsonStore


APPROVAL_PENDING = "pending"
APPROVAL_APPROVED = "approved"
APPROVAL_REJECTED = "rejected"
APPROVAL_NEEDS_DETAILS = "needs_details"


@dataclass(slots=True)
class ApprovalResolutionResult:
    approval: Approval
    approval_path: str
    synced_runs: list[str]
    synced_decisions: list[str]
    synced_sessions: list[str]

    def to_payload(self) -> dict[str, Any]:
        return {
            "approval": self.approval.to_payload(),
            "approval_path": self.approval_path,
            "synced_runs": list(self.synced_runs),
            "synced_decisions": list(self.synced_decisions),
            "synced_sessions": list(self.synced_sessions),
            "executor_called": False,
            "sdep_request_sent": False,
            "executed": False,
        }


def list_approvals(
    store: LocalJsonStore,
    *,
    status: str | None = None,
) -> list[Approval]:
    approvals: list[Approval] = []
    for approval_id in store.list_record_ids("approvals"):
        approval = approval_from_payload(store.load_approval(approval_id))
        if status is None or approval.status == status:
            approvals.append(approval)
    return sorted(approvals, key=lambda item: (item.requested_at, item.approval_id))


def load_approval(store: LocalJsonStore, approval_id: str) -> Approval:
    return approval_from_payload(store.load_approval(approval_id))


def approve_approval(
    store: LocalJsonStore,
    approval_id: str,
    *,
    actor: str = "user",
    reason: str = "",
    now: datetime | None = None,
) -> ApprovalResolutionResult:
    return resolve_approval(
        store,
        approval_id,
        status=APPROVAL_APPROVED,
        actor=actor,
        reason=reason,
        now=now,
    )


def reject_approval(
    store: LocalJsonStore,
    approval_id: str,
    *,
    actor: str = "user",
    reason: str = "",
    now: datetime | None = None,
) -> ApprovalResolutionResult:
    return resolve_approval(
        store,
        approval_id,
        status=APPROVAL_REJECTED,
        actor=actor,
        reason=reason,
        now=now,
    )


def resolve_approval(
    store: LocalJsonStore,
    approval_id: str,
    *,
    status: str,
    actor: str = "user",
    reason: str = "",
    now: datetime | None = None,
) -> ApprovalResolutionResult:
    if status not in {APPROVAL_APPROVED, APPROVAL_REJECTED, APPROVAL_NEEDS_DETAILS}:
        raise ValueError(f"Unsupported approval status transition: {status}")
    approval = load_approval(store, approval_id)
    if approval.status != APPROVAL_PENDING:
        raise ValueError(
            f"Approval {approval_id} is not pending; current status is {approval.status}."
        )

    resolved = Approval(
        approval_id=approval.approval_id,
        decision_id=approval.decision_id,
        candidate_id=approval.candidate_id,
        status=status,
        mode=approval.mode,
        requested_at=approval.requested_at,
        resolved_at=_timestamp(now or datetime.now(timezone.utc)),
        actor=actor or approval.actor,
        prompt=approval.prompt,
        response=status,
        reason=reason,
        execution_allowed=status == APPROVAL_APPROVED,
        metadata={
            **dict(approval.metadata),
            "resolved_by": "spice.runtime.approval_flow",
        },
    )
    approval_path = store.save_approval(resolved.approval_id, resolved.to_payload())
    synced = _sync_references(store, resolved)
    return ApprovalResolutionResult(
        approval=resolved,
        approval_path=str(approval_path),
        synced_runs=synced["runs"],
        synced_decisions=synced["decisions"],
        synced_sessions=synced["sessions"],
    )


def approval_from_payload(payload: dict[str, Any]) -> Approval:
    approval = safe_dataclass_from_payload(Approval, payload)
    if not approval.approval_id:
        raise ValueError("Approval payload missing approval_id.")
    if not approval.decision_id:
        raise ValueError("Approval payload missing decision_id.")
    return approval


def render_approval_list(approvals: list[Approval]) -> str:
    lines = ["SPICE APPROVALS"]
    if not approvals:
        lines.append("- no approvals found")
        return "\n".join(lines)
    for approval in approvals:
        lines.append(
            "- "
            f"{approval.approval_id} "
            f"status={approval.status} "
            f"decision={approval.decision_id} "
            f"candidate={approval.candidate_id or 'none'}"
        )
    return "\n".join(lines)


def render_approval_details(approval: Approval) -> str:
    lines = [
        "SPICE APPROVAL",
        f"approval_id: {approval.approval_id}",
        f"status: {approval.status}",
        f"decision_id: {approval.decision_id}",
        f"candidate_id: {approval.candidate_id or 'none'}",
        f"mode: {approval.mode}",
        f"execution_allowed: {str(bool(approval.execution_allowed)).lower()}",
        f"requested_at: {approval.requested_at or 'unknown'}",
        f"resolved_at: {approval.resolved_at or 'none'}",
    ]
    if approval.prompt:
        lines.extend(["", "PROMPT", approval.prompt])
    if approval.reason:
        lines.extend(["", "REASON", approval.reason])
    lines.extend(
        [
            "",
            "BOUNDARY",
            "- approving only updates the local approval artifact",
            "- no executor is called",
            "- no SDEP request is sent",
        ]
    )
    return "\n".join(lines)


def render_approval_resolution(result: ApprovalResolutionResult) -> str:
    approval = result.approval
    lines = [
        "SPICE APPROVAL UPDATED",
        f"approval_id: {approval.approval_id}",
        f"status: {approval.status}",
        f"execution_allowed: {str(bool(approval.execution_allowed)).lower()}",
        f"decision_id: {approval.decision_id}",
        f"candidate_id: {approval.candidate_id or 'none'}",
        "",
        "BOUNDARY",
        "- executor_called: false",
        "- sdep_request_sent: false",
        "- executed: false",
    ]
    if result.synced_sessions:
        lines.extend(["", "SYNCED SESSIONS"])
        lines.extend(f"- {session_id}" for session_id in result.synced_sessions)
    return "\n".join(lines)


def _sync_references(store: LocalJsonStore, approval: Approval) -> dict[str, list[str]]:
    synced_runs: list[str] = []
    synced_decisions: list[str] = []
    synced_sessions: list[str] = []

    for run_id in store.list_record_ids("runs"):
        payload = store.load_run(run_id)
        if payload.get("approval_id") != approval.approval_id:
            continue
        payload["approval"] = approval.to_payload()
        decision_payload = payload.get("decision")
        if isinstance(decision_payload, dict):
            _replace_nested_approval(decision_payload, approval)
        session_payload = payload.get("session")
        if isinstance(session_payload, dict):
            session_payload["pending_approval_ids"] = [
                item
                for item in _strings(session_payload.get("pending_approval_ids"))
                if item != approval.approval_id
            ]
        store.save_run(run_id, payload)
        synced_runs.append(run_id)

    for decision_id in store.list_record_ids("decisions"):
        payload = store.load_decision(decision_id)
        if _replace_nested_approval(payload, approval):
            store.save_decision(decision_id, payload)
            synced_decisions.append(decision_id)

    for session_id in store.list_record_ids("sessions"):
        payload = store.load_session(session_id)
        session = SessionRecord.from_payload(payload)
        if approval.approval_id not in session.pending_approval_ids:
            continue
        updated = SessionRecord(
            session_id=session.session_id,
            created_at=session.created_at,
            updated_at=approval.resolved_at or session.updated_at,
            status=session.status,
            run_ids=list(session.run_ids),
            decision_ids=list(session.decision_ids),
            approval_ids=list(session.approval_ids),
            active_state_ref=session.active_state_ref,
            last_run_id=session.last_run_id,
            last_decision_id=session.last_decision_id,
            last_trace_ref=session.last_trace_ref,
            pending_approval_ids=[
                item for item in session.pending_approval_ids if item != approval.approval_id
            ],
            metadata=dict(session.metadata),
        )
        store.save_session(updated.session_id, updated.to_payload())
        synced_sessions.append(session_id)

    return {"runs": synced_runs, "decisions": synced_decisions, "sessions": synced_sessions}


def _replace_nested_approval(payload: dict[str, Any], approval: Approval) -> bool:
    changed = False
    approval_payload = approval.to_payload()
    checkpoint = payload.get("checkpoint")
    if isinstance(checkpoint, dict):
        nested = checkpoint.get("approval")
        if isinstance(nested, dict) and nested.get("approval_id") == approval.approval_id:
            checkpoint["approval"] = approval_payload
            changed = True
    nested = payload.get("approval")
    if isinstance(nested, dict) and nested.get("approval_id") == approval.approval_id:
        payload["approval"] = approval_payload
        changed = True
    return changed


def _strings(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str) and item]


def _timestamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).replace(microsecond=0).isoformat()
