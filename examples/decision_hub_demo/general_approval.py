from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from spice.decision.general import GenericCandidate
from spice.decision.general.approval import Approval
from spice.decision.general.types import payload_value

from examples.decision_hub_demo.general_adapter import GeneralDecisionHubResult
from examples.decision_hub_demo.ids import make_confirmation_id, timestamp_segment


@dataclass(slots=True)
class GeneralApprovalBridgeResult:
    """Read-only approval bridge for a General Core decision result."""

    status: str
    decision_id: str
    trace_ref: str
    selected_candidate_id: str
    approval: Approval | None
    confirmation_request: dict[str, Any] | None
    selected_candidate: GenericCandidate
    reason: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "decision_id": self.decision_id,
            "trace_ref": self.trace_ref,
            "selected_candidate_id": self.selected_candidate_id,
            "approval": self.approval.to_payload() if self.approval else None,
            "confirmation_request": payload_value(self.confirmation_request),
            "selected_candidate": self.selected_candidate.to_payload(),
            "reason": self.reason,
            "execution_allowed": False,
            "execution": None,
            "state_updated": False,
        }


def build_general_approval_bridge(
    result: GeneralDecisionHubResult,
    *,
    now: datetime | None = None,
) -> GeneralApprovalBridgeResult:
    """Convert a General decision checkpoint into a pending approval payload.

    This is a bridge into the demo's confirmation shape only. It does not store,
    resolve, authorize, execute, or create an ExecutionIntent.
    """

    selected_candidate_id = result.policy_result.checkpoint.selected_candidate_id
    selected_candidate = _selected_candidate(result, selected_candidate_id)
    approval = result.policy_result.checkpoint.approval

    if not selected_candidate.requires_confirmation:
        return GeneralApprovalBridgeResult(
            status="approval_not_required",
            decision_id=result.policy_result.checkpoint.decision_id,
            trace_ref=result.policy_result.checkpoint.trace_ref,
            selected_candidate_id=selected_candidate_id,
            approval=approval,
            confirmation_request=None,
            selected_candidate=selected_candidate,
            reason="selected candidate does not require approval before execution",
        )

    created = now or datetime.now(timezone.utc).replace(microsecond=0)
    if approval is None:
        approval = Approval(
            approval_id=_approval_id(
                result.policy_result.checkpoint.decision_id,
                selected_candidate_id,
            ),
            decision_id=result.policy_result.checkpoint.decision_id,
            candidate_id=selected_candidate_id,
            status="pending",
            mode="confirm_before_execution",
            requested_at=timestamp_segment(created),
            execution_allowed=False,
        )
    elif not approval.requested_at:
        approval = Approval(
            approval_id=approval.approval_id,
            decision_id=approval.decision_id,
            candidate_id=approval.candidate_id,
            status=approval.status,
            mode=approval.mode,
            requested_at=timestamp_segment(created),
            resolved_at=approval.resolved_at,
            actor=approval.actor,
            prompt=approval.prompt,
            response=approval.response,
            reason=approval.reason,
            execution_allowed=approval.execution_allowed,
            metadata=dict(approval.metadata),
        )

    confirmation_request = _confirmation_request_payload(
        result=result,
        selected_candidate=selected_candidate,
        approval=approval,
        now=created,
    )
    return GeneralApprovalBridgeResult(
        status="approval_required",
        decision_id=result.policy_result.checkpoint.decision_id,
        trace_ref=result.policy_result.checkpoint.trace_ref,
        selected_candidate_id=selected_candidate_id,
        approval=approval,
        confirmation_request=confirmation_request,
        selected_candidate=selected_candidate,
        reason="selected candidate requires approval before crossing the execution boundary",
    )


def build_general_approval_artifact(
    result: GeneralDecisionHubResult,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    created = now or datetime.now(timezone.utc).replace(microsecond=0)
    bridge = build_general_approval_bridge(result, now=created)
    bridge_payload = bridge.to_payload()
    return {
        "path_type": "read_only_general_approval",
        "generated_by": "general_approval_bridge",
        "decision_id": bridge.decision_id,
        "trace_ref": bridge.trace_ref,
        "selected_candidate_id": bridge.selected_candidate_id,
        "approval": bridge.approval.to_payload() if bridge.approval else None,
        "confirmation_request": payload_value(bridge.confirmation_request),
        "execution_allowed": False,
        "execution": None,
        "state_updated": False,
        "created_at": timestamp_segment(created),
        "approval_bridge": bridge_payload,
    }


def _confirmation_request_payload(
    *,
    result: GeneralDecisionHubResult,
    selected_candidate: GenericCandidate,
    approval: Approval,
    now: datetime,
) -> dict[str, Any]:
    checkpoint = result.policy_result.checkpoint
    acted_on = selected_candidate.target_refs[0] if selected_candidate.target_refs else None
    selected_action = selected_candidate.action_type
    confirmation_id = make_confirmation_id(
        now=now,
        decision_id=checkpoint.decision_id,
        selected_action=selected_action,
        acted_on=acted_on,
    )
    return {
        "confirmation_id": confirmation_id,
        "approval_id": approval.approval_id,
        "decision_id": checkpoint.decision_id,
        "trace_ref": checkpoint.trace_ref,
        "candidate_id": selected_candidate.candidate_id,
        "selected_action": selected_action,
        "acted_on": str(acted_on) if acted_on else None,
        "human_summary": f"Approve {selected_action}: {selected_candidate.intent}",
        "reason_summary": list(selected_candidate.why_available),
        "options": [
            {"key": "1", "value": "confirm"},
            {"key": "2", "value": "reject"},
            {"key": "3", "value": "details"},
        ],
        "created_at": timestamp_segment(now),
        "execution_allowed": False,
        "execution": None,
        "metadata": {
            "path_type": "read_only_general_approval",
            "source": "general_approval_bridge",
            "execution_boundary": selected_candidate.execution_boundary.to_payload(),
        },
    }


def _selected_candidate(
    result: GeneralDecisionHubResult,
    selected_candidate_id: str,
) -> GenericCandidate:
    for candidate in result.candidates:
        if candidate.candidate_id == selected_candidate_id:
            return candidate
    raise ValueError(f"selected candidate not found: {selected_candidate_id}")


def _approval_id(decision_id: str, candidate_id: str) -> str:
    digest = sha256(f"{decision_id}\n{candidate_id}".encode("utf-8")).hexdigest()[:12]
    return f"approval.{_id_segment(decision_id)}.{_id_segment(candidate_id)}.{digest}"


def _id_segment(value: str) -> str:
    normalized = "".join(
        char if char.isalnum() else "_"
        for char in str(value).strip().lower()
    ).strip("_")
    return normalized[:48] or "unknown"
