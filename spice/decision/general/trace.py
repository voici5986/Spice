from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from spice.decision.general.approval import Approval
from spice.decision.general.types import PayloadRecord, safe_dataclass_from_payload


@dataclass(slots=True)
class TraceRefs(PayloadRecord):
    decision_id: str
    trace_ref: str
    state_ref: str = ""
    profile_ref: str = ""
    observation_refs: list[str] = field(default_factory=list)
    candidate_refs: list[str] = field(default_factory=list)
    execution_ref: str = ""
    outcome_refs: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CandidateTraceRef(PayloadRecord):
    candidate_id: str
    action_type: str
    status: str = "considered"
    score_ref: str = ""
    veto_ref: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "CandidateTraceRef":
        return cls(
            candidate_id=str(payload.get("candidate_id", "")),
            action_type=str(payload.get("action_type", "unknown")),
            status=str(payload.get("status", "considered")),
            score_ref=str(payload.get("score_ref", "")),
            veto_ref=str(payload.get("veto_ref", "")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class DecisionCheckpoint(PayloadRecord):
    decision_id: str
    trace_ref: str
    state_ref: str
    profile_ref: str
    selected_candidate_id: str = ""
    status: str = "recommended"
    recommendation: str = ""
    candidate_refs: list[CandidateTraceRef] = field(default_factory=list)
    approval: Approval | None = None
    compare_ref: str = ""
    execution_boundary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "DecisionCheckpoint":
        approval_payload = payload.get("approval")
        return cls(
            decision_id=str(payload.get("decision_id", "")),
            trace_ref=str(payload.get("trace_ref", "")),
            state_ref=str(payload.get("state_ref", "")),
            profile_ref=str(payload.get("profile_ref", "")),
            selected_candidate_id=str(payload.get("selected_candidate_id", "")),
            status=str(payload.get("status", "recommended")),
            recommendation=str(payload.get("recommendation", "")),
            candidate_refs=[
                CandidateTraceRef.from_payload(item)
                for item in _list(payload.get("candidate_refs"))
                if isinstance(item, dict)
            ],
            approval=safe_dataclass_from_payload(Approval, approval_payload)
            if isinstance(approval_payload, dict)
            else None,
            compare_ref=str(payload.get("compare_ref", "")),
            execution_boundary=dict(payload.get("execution_boundary", {})),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class DecisionTrace(PayloadRecord):
    trace_ref: str
    decision_id: str
    state_ref: str = ""
    profile_ref: str = ""
    checkpoint_ref: str = ""
    observation_refs: list[str] = field(default_factory=list)
    candidate_refs: list[CandidateTraceRef] = field(default_factory=list)
    selected_candidate_id: str = ""
    approval_ref: str = ""
    execution_ref: str = ""
    outcome_refs: list[str] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def refs(self) -> TraceRefs:
        return TraceRefs(
            decision_id=self.decision_id,
            trace_ref=self.trace_ref,
            state_ref=self.state_ref,
            profile_ref=self.profile_ref,
            observation_refs=list(self.observation_refs),
            candidate_refs=[item.candidate_id for item in self.candidate_refs],
            execution_ref=self.execution_ref,
            outcome_refs=list(self.outcome_refs),
        )


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []
