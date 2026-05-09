from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from typing import Any


def payload_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {str(k): payload_value(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): payload_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [payload_value(item) for item in value]
    if isinstance(value, tuple):
        return [payload_value(item) for item in value]
    return value


def safe_dataclass_from_payload(cls: type[Any], payload: Any) -> Any:
    if not isinstance(payload, dict):
        raise ValueError(f"{cls.__name__} payload must be a dict")
    allowed = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in payload.items() if key in allowed})


@dataclass(slots=True)
class PayloadRecord:
    def to_payload(self) -> dict[str, Any]:
        return payload_value(self)


@dataclass(slots=True)
class Signal(PayloadRecord):
    signal_id: str
    source: str
    kind: str
    summary: str
    subject_ref: str = ""
    observed_at: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Intent(PayloadRecord):
    intent_id: str
    summary: str
    source_signal_refs: list[str] = field(default_factory=list)
    target_refs: list[str] = field(default_factory=list)
    desired_outcome: str = ""
    urgency: str = "unknown"
    status: str = "active"
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Commitment(PayloadRecord):
    commitment_id: str
    title: str
    start_at: str = ""
    end_at: str = ""
    prep_start_at: str = ""
    fixed: bool = True
    priority: str = "normal"
    status: str = "active"
    source_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class WorkItem(PayloadRecord):
    work_item_id: str
    title: str
    status: str = "open"
    urgency: str = "unknown"
    estimate_minutes: int | None = None
    owner: str = ""
    source_refs: list[str] = field(default_factory=list)
    blocker_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Resource(PayloadRecord):
    resource_id: str
    kind: str
    status: str = "available"
    capacity: dict[str, Any] = field(default_factory=dict)
    constraints: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Capability(PayloadRecord):
    capability_id: str
    provider: str
    scope: str
    status: str = "available"
    requires_confirmation: bool = True
    side_effects: list[str] = field(default_factory=list)
    max_duration_seconds: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Constraint(PayloadRecord):
    constraint_id: str
    kind: str
    description: str
    severity: str = "medium"
    applies_to_refs: list[str] = field(default_factory=list)
    status: str = "active"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Risk(PayloadRecord):
    risk_id: str
    kind: str
    description: str
    level: str = "unknown"
    applies_to_refs: list[str] = field(default_factory=list)
    mitigation_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OpenLoop(PayloadRecord):
    open_loop_id: str
    summary: str
    status: str = "open"
    owner: str = ""
    due_at: str = ""
    source_refs: list[str] = field(default_factory=list)
    target_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OutcomeRecord(PayloadRecord):
    outcome_id: str
    decision_id: str = ""
    trace_ref: str | None = None
    candidate_id: str | None = None
    execution_ref: str = ""
    protocol_status: str | None = None
    task_status: str | None = None
    status: str = "observed"
    summary: str = ""
    state_delta: dict[str, Any] = field(default_factory=dict)
    evidence_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
