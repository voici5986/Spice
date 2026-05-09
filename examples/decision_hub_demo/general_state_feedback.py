from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from spice.decision.general import (
    GeneralDecisionState,
    GenericObservation,
    ObservationKind,
    reduce_generic_observations,
)
from spice.decision.general.types import payload_value

from examples.decision_hub_demo.ids import timestamp_segment


GENERAL_STATE_FEEDBACK_ADAPTER = "decision_hub_demo.general_state_feedback"


@dataclass(slots=True)
class GeneralStateFeedbackResult:
    """Applied General outcome observation as a read-only state snapshot."""

    status: str
    decision_id: str
    trace_ref: str
    candidate_id: str
    approval_id: str | None
    execution_id: str
    outcome_id: str
    protocol_status: str
    task_status: str
    state_before: GeneralDecisionState
    state_after: GeneralDecisionState
    outcome_observation: GenericObservation
    state_delta: dict[str, Any]
    reason: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "decision_id": self.decision_id,
            "trace_ref": self.trace_ref,
            "candidate_id": self.candidate_id,
            "approval_id": self.approval_id,
            "execution_id": self.execution_id,
            "outcome_id": self.outcome_id,
            "protocol_status": self.protocol_status,
            "task_status": self.task_status,
            "state_before": self.state_before.to_payload(),
            "state_after": self.state_after.to_payload(),
            "state_before_summary": _state_summary(self.state_before),
            "state_after_summary": _state_summary(self.state_after),
            "outcome_observation": self.outcome_observation.to_payload(),
            "state_delta": payload_value(self.state_delta),
            "applied_effects": ["observations.upsert", "outcomes.upsert"],
            "executor_called": False,
            "executed": False,
            "persisted": False,
            "state_updated": True,
            "state_snapshot_updated": True,
            "update_mode": "read_only_snapshot",
            "reason": self.reason,
        }


def build_general_state_feedback(
    state: GeneralDecisionState,
    outcome_artifact: dict[str, Any],
    *,
    now: datetime | None = None,
) -> GeneralStateFeedbackResult:
    """Apply one General outcome observation to a new state snapshot.

    This adapter only reduces the already-built outcome observation. It does
    not call executors, process live SDEP transport, or persist state.
    """

    created = now or datetime.now(timezone.utc).replace(microsecond=0)
    observation_payload = _require_dict(
        outcome_artifact.get("outcome_observation"),
        "outcome_observation",
    )
    observation = GenericObservation.from_payload(observation_payload)
    _validate_outcome_observation(outcome_artifact, observation)

    state_after = reduce_generic_observations(state, [observation])
    metadata = observation.metadata
    return GeneralStateFeedbackResult(
        status="state_feedback_applied",
        decision_id=_required_string(metadata.get("decision_id"), "decision_id"),
        trace_ref=_required_string(metadata.get("trace_ref"), "trace_ref"),
        candidate_id=_required_string(metadata.get("candidate_id"), "candidate_id"),
        approval_id=_optional_string(metadata.get("approval_id")),
        execution_id=_required_string(metadata.get("execution_id"), "execution_id"),
        outcome_id=_required_string(metadata.get("outcome_id"), "outcome_id"),
        protocol_status=_required_string(metadata.get("protocol_status"), "protocol_status"),
        task_status=_required_string(metadata.get("task_status"), "task_status"),
        state_before=state,
        state_after=state_after,
        outcome_observation=observation,
        state_delta=_state_delta(observation),
        reason="outcome observation reduced into a new GeneralDecisionState snapshot",
    )


def build_general_state_feedback_artifact(
    state: GeneralDecisionState,
    outcome_artifact: dict[str, Any],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    created = now or datetime.now(timezone.utc).replace(microsecond=0)
    result = build_general_state_feedback(state, outcome_artifact, now=created)
    payload = result.to_payload()
    return {
        "path_type": "read_only_general_state_feedback",
        "generated_by": GENERAL_STATE_FEEDBACK_ADAPTER,
        "decision_id": result.decision_id,
        "trace_ref": result.trace_ref,
        "candidate_id": result.candidate_id,
        "approval_id": result.approval_id,
        "execution_id": result.execution_id,
        "outcome_id": result.outcome_id,
        "protocol_status": result.protocol_status,
        "task_status": result.task_status,
        "status": result.status,
        "state_updated": True,
        "state_snapshot_updated": True,
        "update_mode": "read_only_snapshot",
        "persisted": False,
        "executor_called": False,
        "executed": False,
        "execution": None,
        "created_at": timestamp_segment(created),
        "state_before_summary": payload["state_before_summary"],
        "state_after_summary": payload["state_after_summary"],
        "state_delta": payload["state_delta"],
        "applied_effects": payload["applied_effects"],
        "outcome_observation": payload["outcome_observation"],
        "state_before": payload["state_before"],
        "state_after": payload["state_after"],
        "state_feedback": payload,
    }


def _validate_outcome_observation(
    outcome_artifact: dict[str, Any],
    observation: GenericObservation,
) -> None:
    path_type = _required_string(outcome_artifact.get("path_type"), "path_type")
    if path_type != "read_only_general_outcome_return":
        raise ValueError(
            "outcome_artifact.path_type must be 'read_only_general_outcome_return', "
            f"got {path_type!r}"
        )

    kind = observation.kind.value if isinstance(observation.kind, ObservationKind) else str(observation.kind)
    if kind != ObservationKind.OUTCOME.value:
        raise ValueError(f"outcome_observation.kind must be 'outcome', got {kind!r}")

    checks = {
        "decision_id": _required_string(outcome_artifact.get("decision_id"), "decision_id"),
        "trace_ref": _required_string(outcome_artifact.get("trace_ref"), "trace_ref"),
        "candidate_id": _required_string(outcome_artifact.get("candidate_id"), "candidate_id"),
        "approval_id": outcome_artifact.get("approval_id"),
        "execution_id": _required_string(outcome_artifact.get("execution_id"), "execution_id"),
        "outcome_id": _required_string(outcome_artifact.get("outcome_id"), "outcome_id"),
        "request_id": _required_string(outcome_artifact.get("request_id"), "request_id"),
        "protocol_status": _required_string(outcome_artifact.get("protocol_status"), "protocol_status"),
        "task_status": _required_string(outcome_artifact.get("task_status"), "task_status"),
    }
    for key, expected in checks.items():
        actual = observation.metadata.get(key)
        if expected is not None and actual != expected:
            raise ValueError(
                f"outcome observation metadata.{key} mismatch: "
                f"expected {expected!r}, got {actual!r}"
            )
    outcome_record = outcome_artifact.get("outcome_record")
    if isinstance(outcome_record, dict):
        record_outcome_id = outcome_record.get("outcome_id")
        if record_outcome_id != checks["outcome_id"]:
            raise ValueError(
                "outcome_record.outcome_id mismatch: "
                f"expected {checks['outcome_id']!r}, got {record_outcome_id!r}"
            )


def _state_summary(state: GeneralDecisionState) -> dict[str, Any]:
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
        "approval_count": len(state.approvals),
        "outcome_count": len(state.outcomes),
    }


def _state_delta(observation: GenericObservation) -> dict[str, Any]:
    state_delta = observation.attributes.get("state_delta")
    return dict(state_delta) if isinstance(state_delta, dict) else {}


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} is required")
    return value


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")
    return value


def _optional_string(value: Any) -> str | None:
    return value if isinstance(value, str) and value.strip() else None
