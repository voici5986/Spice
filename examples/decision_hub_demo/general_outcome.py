from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from typing import Any

from spice.decision.general import (
    GenericObservation,
    ObservationEvidence,
    ObservationKind,
    ObservationSource,
    ObservationSubject,
    OutcomeRecord,
)
from spice.decision.general.types import payload_value
from spice.protocols.sdep import SDEPExecuteResponse

from examples.decision_hub_demo.ids import timestamp_segment


GENERAL_OUTCOME_ADAPTER = "decision_hub_demo.general_outcome_adapter"


@dataclass(slots=True)
class GeneralOutcomeReturnResult:
    """Read-only outcome return for a planned General execution handoff."""

    status: str
    decision_id: str
    trace_ref: str
    candidate_id: str
    approval_id: str | None
    execution_id: str
    request_id: str
    protocol_status: str
    task_status: str
    outcome_record: OutcomeRecord
    outcome_observation: GenericObservation
    sdep_response: dict[str, Any]
    reason: str = ""

    def to_payload(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "decision_id": self.decision_id,
            "trace_ref": self.trace_ref,
            "candidate_id": self.candidate_id,
            "approval_id": self.approval_id,
            "execution_id": self.execution_id,
            "request_id": self.request_id,
            "protocol_status": self.protocol_status,
            "task_status": self.task_status,
            "outcome_record": self.outcome_record.to_payload(),
            "outcome_observation": self.outcome_observation.to_payload(),
            "sdep_response": payload_value(self.sdep_response),
            "response_processed": True,
            "executor_called": False,
            "state_updated": False,
            "reason": self.reason,
        }


def build_general_sdep_response_fixture(
    execution_artifact: dict[str, Any],
    *,
    now: datetime | None = None,
    response_status: str = "success",
    task_status: str = "success",
    output: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic local SDEP response fixture for the read-only demo.

    This function does not call an executor. It creates a response-shaped payload
    so the outcome adapter can demonstrate attribution and status handling.
    """

    created = now or datetime.now(timezone.utc).replace(microsecond=0)
    sdep_request = _require_dict(execution_artifact.get("sdep_request"), "sdep_request")
    execution_id = _required_string(execution_artifact.get("execution_id"), "execution_id")
    decision_id = _required_string(execution_artifact.get("decision_id"), "decision_id")
    trace_ref = _required_string(execution_artifact.get("trace_ref"), "trace_ref")
    candidate_id = _required_string(execution_artifact.get("candidate_id"), "candidate_id")
    approval_id = _optional_string(execution_artifact.get("approval_id"))
    request_id = _required_string(sdep_request.get("request_id"), "sdep_request.request_id")

    response = {
        "protocol": "sdep",
        "sdep_version": "0.1",
        "message_type": "execute.response",
        "message_id": f"sdep-msg.general.response.{_hash(execution_id)}",
        "request_id": request_id,
        "timestamp": timestamp_segment(created),
        "responder": {
            "id": "agent.general.fixture",
            "name": "General Fixture Executor",
            "version": "0.1",
            "vendor": "Spice",
            "implementation": "read-only-fixture",
            "role": "executor",
        },
        "status": response_status,
        "outcome": {
            "execution_id": execution_id,
            "status": task_status,
            "outcome_type": "observation",
            "output": output
            or {
                "summary": "Planned action completed by read-only fixture.",
                "state_delta": {
                    "updated_refs": [candidate_id],
                    "task_status": task_status,
                },
            },
            "artifacts": [],
            "metrics": {},
            "metadata": {
                "adapter": GENERAL_OUTCOME_ADAPTER,
                "fixture": True,
            },
        },
        "traceability": {
            "execution_id": execution_id,
            "spice_decision_id": decision_id,
            "trace_ref": trace_ref,
            "candidate_id": candidate_id,
            "approval_id": approval_id,
        },
        "metadata": {
            "adapter": GENERAL_OUTCOME_ADAPTER,
            "fixture": True,
            "planning_only": False,
        },
    }
    parsed = SDEPExecuteResponse.from_dict(response)
    return parsed.to_dict()


def build_general_outcome_return(
    execution_artifact: dict[str, Any],
    response_payload: dict[str, Any] | SDEPExecuteResponse,
    *,
    now: datetime | None = None,
) -> GeneralOutcomeReturnResult:
    """Convert one SDEP execute.response into a General outcome observation.

    This is a read-only adapter. It validates attribution and produces outcome
    records only; it does not apply them to GeneralDecisionState.
    """

    created = now or datetime.now(timezone.utc).replace(microsecond=0)
    response = (
        response_payload
        if isinstance(response_payload, SDEPExecuteResponse)
        else SDEPExecuteResponse.from_dict(response_payload)
    )
    response_dict = response.to_dict()

    decision_id = _required_string(execution_artifact.get("decision_id"), "decision_id")
    trace_ref = _required_string(execution_artifact.get("trace_ref"), "trace_ref")
    candidate_id = _required_string(
        execution_artifact.get("candidate_id") or execution_artifact.get("selected_candidate_id"),
        "candidate_id",
    )
    approval_id = _optional_string(execution_artifact.get("approval_id"))
    execution_id = _required_string(execution_artifact.get("execution_id"), "execution_id")
    sdep_request = _require_dict(execution_artifact.get("sdep_request"), "sdep_request")
    request_id = _required_string(sdep_request.get("request_id"), "sdep_request.request_id")

    _validate_response_attribution(
        response=response,
        request_id=request_id,
        decision_id=decision_id,
        trace_ref=trace_ref,
        candidate_id=candidate_id,
        approval_id=approval_id,
        execution_id=execution_id,
    )

    outcome_record = _outcome_record_from_response(
        response=response,
        decision_id=decision_id,
        trace_ref=trace_ref,
        candidate_id=candidate_id,
        execution_id=execution_id,
        request_id=request_id,
    )
    outcome_observation = _outcome_observation_from_record(
        response=response,
        outcome=outcome_record,
        approval_id=approval_id,
        request_id=request_id,
        now=created,
    )
    return GeneralOutcomeReturnResult(
        status="outcome_observed",
        decision_id=decision_id,
        trace_ref=trace_ref,
        candidate_id=candidate_id,
        approval_id=approval_id,
        execution_id=execution_id,
        request_id=request_id,
        protocol_status=response.status,
        task_status=response.outcome.status,
        outcome_record=outcome_record,
        outcome_observation=outcome_observation,
        sdep_response=response_dict,
        reason="SDEP execute.response converted into a General outcome observation",
    )


def build_general_outcome_artifact(
    execution_artifact: dict[str, Any],
    response_payload: dict[str, Any] | SDEPExecuteResponse,
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    created = now or datetime.now(timezone.utc).replace(microsecond=0)
    result = build_general_outcome_return(
        execution_artifact,
        response_payload,
        now=created,
    )
    result_payload = result.to_payload()
    return {
        "path_type": "read_only_general_outcome_return",
        "generated_by": GENERAL_OUTCOME_ADAPTER,
        "decision_id": result.decision_id,
        "trace_ref": result.trace_ref,
        "candidate_id": result.candidate_id,
        "approval_id": result.approval_id,
        "execution_id": result.execution_id,
        "request_id": result.request_id,
        "status": result.status,
        "outcome_id": result.outcome_record.outcome_id,
        "protocol_status": result.protocol_status,
        "task_status": result.task_status,
        "response_processed": True,
        "executor_called": False,
        "executed": False,
        "execution": None,
        "state_updated": False,
        "created_at": timestamp_segment(created),
        "outcome": result_payload["outcome_record"],
        "outcome_record": result_payload["outcome_record"],
        "outcome_observation": result_payload["outcome_observation"],
        "sdep_response": result_payload["sdep_response"],
        "outcome_return": result_payload,
    }


def _validate_response_attribution(
    *,
    response: SDEPExecuteResponse,
    request_id: str,
    decision_id: str,
    trace_ref: str,
    candidate_id: str,
    approval_id: str | None,
    execution_id: str,
) -> None:
    if response.request_id != request_id:
        raise ValueError(
            f"response request_id mismatch: expected {request_id!r}, got {response.request_id!r}"
        )
    if response.outcome.execution_id != execution_id:
        raise ValueError(
            f"response execution_id mismatch: expected {execution_id!r}, "
            f"got {response.outcome.execution_id!r}"
        )
    traceability = response.traceability
    _require_traceability(traceability, "execution_id", execution_id)
    _require_traceability(traceability, "spice_decision_id", decision_id)
    _require_traceability(traceability, "trace_ref", trace_ref)
    _require_traceability(traceability, "candidate_id", candidate_id)
    if approval_id is not None:
        _require_traceability(traceability, "approval_id", approval_id)


def _outcome_record_from_response(
    *,
    response: SDEPExecuteResponse,
    decision_id: str,
    trace_ref: str,
    candidate_id: str,
    execution_id: str,
    request_id: str,
) -> OutcomeRecord:
    summary = _summary_from_response(response)
    return OutcomeRecord(
        outcome_id=f"outcome.{_hash(f'{request_id}\\n{execution_id}\\n{response.status}\\n{response.outcome.status}')}",
        decision_id=decision_id,
        trace_ref=trace_ref,
        candidate_id=candidate_id,
        execution_ref=execution_id,
        protocol_status=response.status,
        task_status=response.outcome.status,
        status="observed",
        summary=summary,
        state_delta=_state_delta(response.outcome.output),
        evidence_refs=[response.message_id],
        metadata={
            "adapter": GENERAL_OUTCOME_ADAPTER,
            "approval_id": response.traceability.get("approval_id"),
            "execution_id": execution_id,
            "request_id": request_id,
            "response_message_id": response.message_id,
            "protocol_status": response.status,
            "task_status": response.outcome.status,
            "traceability": dict(response.traceability),
            "responder": response.responder.to_dict(),
            "outcome_type": response.outcome.outcome_type,
            "output": dict(response.outcome.output),
            "artifacts": list(response.outcome.artifacts),
            "metrics": dict(response.outcome.metrics),
            "error": response.error.to_dict() if response.error else None,
        },
    )


def _outcome_observation_from_record(
    *,
    response: SDEPExecuteResponse,
    outcome: OutcomeRecord,
    approval_id: str | None,
    request_id: str,
    now: datetime,
) -> GenericObservation:
    refs = [
        outcome.decision_id,
        outcome.trace_ref or "",
        outcome.candidate_id or "",
        approval_id or "",
        outcome.execution_ref,
        request_id,
    ]
    return GenericObservation(
        observation_id=f"obs.{outcome.outcome_id}",
        kind=ObservationKind.OUTCOME,
        source=ObservationSource(
            provider=response.responder.id,
            channel="sdep",
            external_id=response.message_id,
            actor=response.responder.name,
            received_at=timestamp_segment(now),
            metadata={"adapter": GENERAL_OUTCOME_ADAPTER},
        ),
        subject=ObservationSubject(
            subject_id=outcome.execution_ref,
            subject_type="execution",
            title=outcome.summary,
            refs=[ref for ref in refs if ref],
        ),
        summary=outcome.summary,
        attributes=outcome.to_payload(),
        evidence=[
            ObservationEvidence(
                evidence_id=response.message_id,
                kind="sdep.execute.response",
                summary=f"SDEP response status={response.status}, task={response.outcome.status}",
                metadata={
                    "request_id": request_id,
                    "message_type": response.message_type,
                },
            )
        ],
        refs=[ref for ref in refs if ref],
        metadata={
            "path_type": "read_only_general_outcome_return",
            "source": GENERAL_OUTCOME_ADAPTER,
            "decision_id": outcome.decision_id,
            "trace_ref": outcome.trace_ref,
            "candidate_id": outcome.candidate_id,
            "approval_id": approval_id,
            "execution_id": outcome.execution_ref,
            "outcome_id": outcome.outcome_id,
            "request_id": request_id,
            "response_message_id": response.message_id,
            "protocol_status": response.status,
            "task_status": response.outcome.status,
            "traceability": dict(response.traceability),
            "responder": response.responder.to_dict(),
            "outcome_type": response.outcome.outcome_type,
            "output": dict(response.outcome.output),
            "artifacts": list(response.outcome.artifacts),
            "metrics": dict(response.outcome.metrics),
            "error": response.error.to_dict() if response.error else None,
        },
    )


def _summary_from_response(response: SDEPExecuteResponse) -> str:
    output_summary = response.outcome.output.get("summary")
    if output_summary:
        return str(output_summary)
    if response.error is not None:
        return response.error.message
    return f"SDEP response {response.status}; task {response.outcome.status}."


def _state_delta(output: dict[str, Any]) -> dict[str, Any]:
    state_delta = output.get("state_delta")
    return dict(state_delta) if isinstance(state_delta, dict) else {}


def _require_traceability(
    traceability: dict[str, Any],
    key: str,
    expected: str,
) -> None:
    actual = traceability.get(key)
    if actual != expected:
        raise ValueError(
            f"response traceability.{key} mismatch: expected {expected!r}, got {actual!r}"
        )


def _required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} is required")
    return value.strip()


def _optional_string(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_dict(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object")
    return dict(value)


def _hash(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()[:12]
