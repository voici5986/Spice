from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from spice.decision.general.types import PayloadRecord
from spice.protocols.observation import Observation


class ObservationKind(str, Enum):
    SIGNAL = "signal"
    INTENT = "intent"
    COMMITMENT = "commitment"
    WORK_ITEM = "work_item"
    CAPABILITY = "capability"
    CONSTRAINT = "constraint"
    RISK = "risk"
    OUTCOME = "outcome"
    USER_REPLY = "user_reply"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class ObservationSource(PayloadRecord):
    provider: str
    channel: str = ""
    external_id: str = ""
    actor: str = ""
    received_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ObservationSource":
        return cls(
            provider=str(payload.get("provider", "unknown")),
            channel=str(payload.get("channel", "")),
            external_id=str(payload.get("external_id", "")),
            actor=str(payload.get("actor", "")),
            received_at=str(payload.get("received_at", "")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class ObservationSubject(PayloadRecord):
    subject_id: str
    subject_type: str
    title: str = ""
    refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ObservationSubject":
        return cls(
            subject_id=str(payload.get("subject_id", "")),
            subject_type=str(payload.get("subject_type", "unknown")),
            title=str(payload.get("title", "")),
            refs=[str(item) for item in _list(payload.get("refs"))],
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class ObservationEvidence(PayloadRecord):
    evidence_id: str
    kind: str
    summary: str = ""
    content: str = ""
    uri: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ObservationEvidence":
        return cls(
            evidence_id=str(payload.get("evidence_id", "")),
            kind=str(payload.get("kind", "unknown")),
            summary=str(payload.get("summary", "")),
            content=str(payload.get("content", "")),
            uri=str(payload.get("uri", "")),
            metadata=dict(payload.get("metadata", {})),
        )


@dataclass(slots=True)
class ObservationConfidence(PayloadRecord):
    score: float | None = None
    level: str = "unknown"
    uncertain_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @classmethod
    def from_payload(cls, payload: dict[str, Any] | None) -> "ObservationConfidence":
        if not isinstance(payload, dict):
            return cls()
        score_raw = payload.get("score")
        try:
            score = None if score_raw is None else float(score_raw)
        except (TypeError, ValueError):
            score = None
        return cls(
            score=score,
            level=str(payload.get("level", "unknown")),
            uncertain_fields=[str(item) for item in _list(payload.get("uncertain_fields"))],
            missing_fields=[str(item) for item in _list(payload.get("missing_fields"))],
            notes=[str(item) for item in _list(payload.get("notes"))],
        )


@dataclass(slots=True)
class GenericObservation(PayloadRecord):
    observation_id: str
    kind: ObservationKind | str
    source: ObservationSource
    subject: ObservationSubject | None = None
    summary: str = ""
    attributes: dict[str, Any] = field(default_factory=dict)
    evidence: list[ObservationEvidence] = field(default_factory=list)
    confidence: ObservationConfidence = field(default_factory=ObservationConfidence)
    refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "GenericObservation":
        source_payload = payload.get("source", {})
        subject_payload = payload.get("subject")
        return cls(
            observation_id=str(payload.get("observation_id", "")),
            kind=str(payload.get("kind", ObservationKind.UNKNOWN.value)),
            source=ObservationSource.from_payload(
                source_payload if isinstance(source_payload, dict) else {}
            ),
            subject=(
                ObservationSubject.from_payload(subject_payload)
                if isinstance(subject_payload, dict)
                else None
            ),
            summary=str(payload.get("summary", "")),
            attributes=dict(payload.get("attributes", {})),
            evidence=[
                ObservationEvidence.from_payload(item)
                for item in _list(payload.get("evidence"))
                if isinstance(item, dict)
            ],
            confidence=ObservationConfidence.from_payload(payload.get("confidence")),
            refs=[str(item) for item in _list(payload.get("refs"))],
            metadata=dict(payload.get("metadata", {})),
        )

    @classmethod
    def from_protocol_observation(cls, observation: Observation) -> "GenericObservation":
        source_name = observation.source or "unknown"
        attributes = dict(observation.attributes)
        subject_id = str(attributes.get("subject_id", ""))
        subject_type = str(attributes.get("subject_type", observation.observation_type))
        subject = None
        if subject_id:
            subject = ObservationSubject(
                subject_id=subject_id,
                subject_type=subject_type,
                title=str(attributes.get("title", "")),
            )
        return cls(
            observation_id=observation.id,
            kind=observation.observation_type or ObservationKind.UNKNOWN,
            source=ObservationSource(provider=source_name),
            subject=subject,
            summary=str(attributes.get("summary", "")),
            attributes=attributes,
            refs=list(observation.refs),
            metadata=dict(observation.metadata),
        )


def _list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    return []
