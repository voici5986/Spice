from __future__ import annotations

from typing import Any, Callable, TypeVar

from spice.decision.general.observations import GenericObservation, ObservationKind
from spice.decision.general.state import GeneralDecisionState
from spice.decision.general.types import (
    Capability,
    Commitment,
    Constraint,
    Intent,
    OpenLoop,
    OutcomeRecord,
    Resource,
    Risk,
    Signal,
    WorkItem,
)

T = TypeVar("T")


def reduce_generic_observations(
    state: GeneralDecisionState,
    observations: list[GenericObservation],
) -> GeneralDecisionState:
    reduced = _copy_state(state)
    for observation in observations:
        reduced = apply_generic_observation(reduced, observation)
    return reduced


def apply_generic_observation(
    state: GeneralDecisionState,
    observation: GenericObservation,
) -> GeneralDecisionState:
    reduced = _copy_state(state)
    reduced.observations = _upsert(
        reduced.observations,
        observation,
        lambda item: item.observation_id,
    )

    kind = _kind(observation)
    if kind == ObservationKind.SIGNAL.value:
        reduced.signals = _upsert(
            reduced.signals,
            _signal_from_observation(observation),
            lambda item: item.signal_id,
        )
    elif kind == ObservationKind.INTENT.value:
        reduced.intents = _upsert(
            reduced.intents,
            _intent_from_observation(observation),
            lambda item: item.intent_id,
        )
    elif kind == ObservationKind.COMMITMENT.value:
        reduced.commitments = _upsert(
            reduced.commitments,
            _commitment_from_observation(observation),
            lambda item: item.commitment_id,
        )
    elif kind == ObservationKind.WORK_ITEM.value:
        reduced.work_items = _upsert(
            reduced.work_items,
            _work_item_from_observation(observation),
            lambda item: item.work_item_id,
        )
    elif kind == ObservationKind.CAPABILITY.value:
        reduced.capabilities = _upsert(
            reduced.capabilities,
            _capability_from_observation(observation),
            lambda item: item.capability_id,
        )
    elif kind == ObservationKind.CONSTRAINT.value:
        reduced.constraints = _upsert(
            reduced.constraints,
            _constraint_from_observation(observation),
            lambda item: item.constraint_id,
        )
    elif kind == ObservationKind.RISK.value:
        reduced.risks = _upsert(
            reduced.risks,
            _risk_from_observation(observation),
            lambda item: item.risk_id,
        )
    elif kind == ObservationKind.OUTCOME.value:
        reduced.outcomes = _upsert(
            reduced.outcomes,
            _outcome_from_observation(observation),
            lambda item: item.outcome_id,
        )
    elif kind == "resource":
        reduced.resources = _upsert(
            reduced.resources,
            _resource_from_observation(observation),
            lambda item: item.resource_id,
        )
    elif kind == "open_loop":
        reduced.open_loops = _upsert(
            reduced.open_loops,
            _open_loop_from_observation(observation),
            lambda item: item.open_loop_id,
        )
    return reduced


def _copy_state(state: GeneralDecisionState) -> GeneralDecisionState:
    return GeneralDecisionState(
        state_id=state.state_id,
        schema_version=state.schema_version,
        signals=list(state.signals),
        observations=list(state.observations),
        intents=list(state.intents),
        commitments=list(state.commitments),
        work_items=list(state.work_items),
        resources=list(state.resources),
        capabilities=list(state.capabilities),
        constraints=list(state.constraints),
        risks=list(state.risks),
        open_loops=list(state.open_loops),
        approvals=list(state.approvals),
        decision_checkpoints=list(state.decision_checkpoints),
        outcomes=list(state.outcomes),
        trace_refs=list(state.trace_refs),
        metadata=dict(state.metadata),
    )


def _signal_from_observation(observation: GenericObservation) -> Signal:
    attrs = observation.attributes
    return Signal(
        signal_id=_id(observation, "signal_id", "signal"),
        source=_string(attrs.get("source")) or observation.source.provider,
        kind=_string(attrs.get("kind")) or _kind(observation),
        summary=_summary(observation),
        subject_ref=_subject_id(observation),
        observed_at=_string(attrs.get("observed_at")) or observation.source.received_at,
        payload=dict(attrs),
        confidence=observation.confidence.score,
        refs=_refs(observation),
        metadata=_metadata(observation),
    )


def _intent_from_observation(observation: GenericObservation) -> Intent:
    attrs = observation.attributes
    return Intent(
        intent_id=_id(observation, "intent_id", "intent"),
        summary=_summary(observation),
        source_signal_refs=_source_refs(observation),
        target_refs=_target_refs(observation),
        desired_outcome=_string(attrs.get("desired_outcome")),
        urgency=_string(attrs.get("urgency")) or "unknown",
        status=_string(attrs.get("status")) or "active",
        confidence=observation.confidence.score,
        metadata=_metadata(observation),
    )


def _commitment_from_observation(observation: GenericObservation) -> Commitment:
    attrs = observation.attributes
    return Commitment(
        commitment_id=_id(observation, "commitment_id", "commitment"),
        title=_title(observation),
        start_at=_string(attrs.get("start_at")),
        end_at=_string(attrs.get("end_at")),
        prep_start_at=_string(attrs.get("prep_start_at")),
        fixed=_bool(attrs.get("fixed"), True),
        priority=_string(attrs.get("priority")) or "normal",
        status=_string(attrs.get("status")) or "active",
        source_refs=_source_refs(observation),
        metadata=_metadata(observation),
    )


def _work_item_from_observation(observation: GenericObservation) -> WorkItem:
    attrs = observation.attributes
    return WorkItem(
        work_item_id=_id(observation, "work_item_id", "work_item"),
        title=_title(observation),
        status=_string(attrs.get("status")) or "open",
        urgency=_string(attrs.get("urgency")) or "unknown",
        estimate_minutes=_int_or_none(
            attrs.get("estimate_minutes", attrs.get("estimated_minutes"))
        ),
        owner=_string(attrs.get("owner")),
        source_refs=_source_refs(observation),
        blocker_refs=_string_list(attrs.get("blocker_refs")),
        metadata=_metadata(observation),
    )


def _resource_from_observation(observation: GenericObservation) -> Resource:
    attrs = observation.attributes
    return Resource(
        resource_id=_id(observation, "resource_id", "resource"),
        kind=_string(attrs.get("resource_kind")) or _subject_type(observation) or "unknown",
        status=_string(attrs.get("status")) or "available",
        capacity=_dict(attrs.get("capacity")),
        constraints=_string_list(attrs.get("constraints")),
        metadata=_metadata(observation),
    )


def _capability_from_observation(observation: GenericObservation) -> Capability:
    attrs = observation.attributes
    return Capability(
        capability_id=_id(observation, "capability_id", "capability"),
        provider=_string(attrs.get("provider")) or observation.source.provider,
        scope=_string(attrs.get("scope")) or _subject_type(observation) or "general",
        status=_string(attrs.get("status")) or "available",
        requires_confirmation=_bool(attrs.get("requires_confirmation"), True),
        side_effects=_string_list(attrs.get("side_effects")),
        max_duration_seconds=_int_or_none(attrs.get("max_duration_seconds")),
        metadata=_metadata(observation),
    )


def _constraint_from_observation(observation: GenericObservation) -> Constraint:
    attrs = observation.attributes
    return Constraint(
        constraint_id=_id(observation, "constraint_id", "constraint"),
        kind=_string(attrs.get("constraint_kind")) or _subject_type(observation) or "generic",
        description=_string(attrs.get("description")) or _summary(observation),
        severity=_string(attrs.get("severity")) or "medium",
        applies_to_refs=_target_refs(observation),
        status=_string(attrs.get("status")) or "active",
        metadata=_metadata(observation),
    )


def _risk_from_observation(observation: GenericObservation) -> Risk:
    attrs = observation.attributes
    return Risk(
        risk_id=_id(observation, "risk_id", "risk"),
        kind=_string(attrs.get("risk_kind")) or _subject_type(observation) or "generic",
        description=_string(attrs.get("description")) or _summary(observation),
        level=_string(attrs.get("level")) or "unknown",
        applies_to_refs=_target_refs(observation),
        mitigation_refs=_string_list(attrs.get("mitigation_refs")),
        metadata=_metadata(observation),
    )


def _open_loop_from_observation(observation: GenericObservation) -> OpenLoop:
    attrs = observation.attributes
    return OpenLoop(
        open_loop_id=_id(observation, "open_loop_id", "open_loop"),
        summary=_summary(observation),
        status=_string(attrs.get("status")) or "open",
        owner=_string(attrs.get("owner")),
        due_at=_string(attrs.get("due_at")),
        source_refs=_source_refs(observation),
        target_refs=_target_refs(observation),
        metadata=_metadata(observation),
    )


def _outcome_from_observation(observation: GenericObservation) -> OutcomeRecord:
    attrs = observation.attributes
    return OutcomeRecord(
        outcome_id=_id(observation, "outcome_id", "outcome"),
        decision_id=_string(attrs.get("decision_id")),
        trace_ref=_optional_string(attrs.get("trace_ref")),
        candidate_id=_optional_string(attrs.get("candidate_id")),
        execution_ref=_string(attrs.get("execution_ref")),
        protocol_status=_optional_string(attrs.get("protocol_status")),
        task_status=_optional_string(attrs.get("task_status")),
        status=_string(attrs.get("status")) or "observed",
        summary=_summary(observation),
        state_delta=_dict(attrs.get("state_delta")),
        evidence_refs=[item.evidence_id for item in observation.evidence],
        metadata=_metadata(observation),
    )


def _upsert(items: list[T], item: T, key: Callable[[T], str]) -> list[T]:
    item_key = key(item)
    result = [existing for existing in items if key(existing) != item_key]
    result.append(item)
    return result


def _kind(observation: GenericObservation) -> str:
    value = observation.kind
    if isinstance(value, ObservationKind):
        return value.value
    return str(value)


def _id(observation: GenericObservation, attr_name: str, prefix: str) -> str:
    explicit = _string(observation.attributes.get(attr_name))
    if explicit:
        return explicit
    subject_id = _subject_id(observation)
    if subject_id:
        return subject_id
    return f"{prefix}.{observation.observation_id}"


def _title(observation: GenericObservation) -> str:
    explicit = _string(observation.attributes.get("title"))
    if explicit:
        return explicit
    if observation.subject and observation.subject.title:
        return observation.subject.title
    return _summary(observation)


def _summary(observation: GenericObservation) -> str:
    return observation.summary or _string(observation.attributes.get("summary")) or _subject_id(observation) or observation.observation_id


def _target_refs(observation: GenericObservation) -> list[str]:
    refs = _string_list(observation.attributes.get("target_refs"))
    if refs:
        return refs
    if observation.subject and observation.subject.refs:
        return list(observation.subject.refs)
    subject_id = _subject_id(observation)
    return [subject_id] if subject_id else []


def _source_refs(observation: GenericObservation) -> list[str]:
    return [observation.observation_id, *observation.refs]


def _refs(observation: GenericObservation) -> list[str]:
    return [*_source_refs(observation), *_target_refs(observation)]


def _metadata(observation: GenericObservation) -> dict[str, Any]:
    return {
        **dict(observation.metadata),
        "observation_id": observation.observation_id,
        "observation_kind": _kind(observation),
        "source": observation.source.to_payload(),
    }


def _subject_id(observation: GenericObservation) -> str:
    return observation.subject.subject_id if observation.subject else ""


def _subject_type(observation: GenericObservation) -> str:
    return observation.subject.subject_type if observation.subject else ""


def _string(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _optional_string(value: Any) -> str | None:
    text = _string(value)
    return text or None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _dict(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return dict(value)


def _bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
    return default


def _int_or_none(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
