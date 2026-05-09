from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from spice.decision.compare import render_compare_text
from spice.decision import (
    DecisionGuidance,
    HardConstraintGuidance,
    PrimaryObjectiveGuidance,
)
from spice.decision.general import (
    GeneralDecisionState,
    GenericCandidate,
    GenericObservation,
    GenericPolicyAdapter,
    GenericPolicyResult,
    ObservationConfidence,
    ObservationKind,
    ObservationSource,
    ObservationSubject,
    generate_generic_candidates,
    reduce_generic_observations,
)
from spice.protocols.observation import Observation

from examples.decision_hub_demo.state import isoformat_utc, parse_time, stable_slug


@dataclass(slots=True)
class GeneralDecisionHubResult:
    """Read-only general decision path result for the decision hub demo."""

    observations: list[GenericObservation]
    state: GeneralDecisionState
    candidates: list[GenericCandidate]
    policy_result: GenericPolicyResult

    def to_payload(self) -> dict[str, Any]:
        return {
            "observations": [observation.to_payload() for observation in self.observations],
            "state": self.state.to_payload(),
            "candidates": [candidate.to_payload() for candidate in self.candidates],
            "policy_result": self.policy_result.to_payload(),
            "compare_payload": self.policy_result.compare_payload,
        }


def build_demo_protocol_observations(now: datetime) -> list[Observation]:
    """Return the demo provider events used by the general read-only path."""

    return [
        Observation(
            id="obs.demo.capability.codex",
            timestamp=now,
            observation_type="executor_capability_observed",
            source="hermes",
            metadata={
                "adapter": "hermes_capability.v1",
                "reported_by": "hermes",
                "notes": "Codex available via Hermes terminal/codex skill.",
            },
            attributes={
                "capability_id": "cap.external_executor.codex",
                "action_type": "delegate_to_executor",
                "executor": "codex",
                "supported_scopes": ["triage", "review_summary"],
                "requires_confirmation": True,
                "reversible": True,
                "default_time_budget_minutes": 10,
                "availability": "available",
            },
        ),
        Observation(
            id="obs.demo.commitment",
            timestamp=now,
            observation_type="commitment_declared",
            source="whatsapp",
            attributes={
                "commitment_id": "commitment.demo.flight",
                "summary": "Leave for fixed commitment",
                "start_time": (now + timedelta(minutes=42)).isoformat(),
                "end_time": (now + timedelta(minutes=102)).isoformat(),
                "duration_minutes": 60,
                "prep_start_time": (now + timedelta(minutes=12)).isoformat(),
                "priority_hint": "high",
                "flexibility_hint": "fixed",
                "constraint_hints": ["do_not_be_late"],
            },
        ),
        Observation(
            id="obs.demo.github.pr",
            timestamp=now,
            observation_type="work_item_opened",
            source="github",
            attributes={
                "kind": "pull_request",
                "repo": "Dyalwayshappy/Spice",
                "item_id": "123",
                "title": "Fix decision guidance validation",
                "url": "https://github.com/Dyalwayshappy/Spice/pull/123",
                "action": "opened",
                "urgency_hint": "medium",
                "estimated_minutes_hint": 30,
                "requires_attention": True,
                "event_key": "github:Dyalwayshappy/Spice:pull_request:123:opened",
            },
        ),
    ]


def build_general_observations(
    now: datetime,
    protocol_observations: list[Observation] | None = None,
) -> list[GenericObservation]:
    """Normalize demo provider events into general observations.

    This is intentionally a read-only adapter. It does not call the old demo
    reducer, candidate registry, runtime service, SDEP, or executor path.
    """

    protocol_events = list(protocol_observations or build_demo_protocol_observations(now))
    observations: list[GenericObservation] = []
    for event in protocol_events:
        observations.extend(protocol_observation_to_generic(event))
    observations.extend(_derived_constraint_observations(protocol_events, now))
    return observations


def protocol_observation_to_generic(observation: Observation) -> list[GenericObservation]:
    if observation.observation_type == "executor_capability_observed":
        return [_capability_observation(observation)]
    if observation.observation_type == "commitment_declared":
        return [_commitment_observation(observation)]
    if observation.observation_type == "work_item_opened":
        return [
            _work_item_observation(observation),
            _intent_observation_for_work_item(observation),
        ]
    return [_unknown_observation(observation)]


def run_general_read_only_path(
    *,
    now: datetime,
    guidance: DecisionGuidance | None = None,
) -> GeneralDecisionHubResult:
    observations = build_general_observations(now)
    state = reduce_generic_observations(
        GeneralDecisionState(state_id="world.decision_hub_demo.general"),
        observations,
    )
    candidates = generate_generic_candidates(state)
    policy_result = GenericPolicyAdapter(
        guidance or decision_hub_general_guidance()
    ).evaluate(
        state,
        candidates=candidates,
        decision_id=f"decision.general.decision_hub.{_timestamp_slug(now)}",
        trace_ref=f"trace.general.decision_hub.{_timestamp_slug(now)}",
    )
    return GeneralDecisionHubResult(
        observations=observations,
        state=state,
        candidates=candidates,
        policy_result=policy_result,
    )


def build_general_compare_artifact(*, now: datetime) -> dict[str, Any]:
    return run_general_read_only_path(now=now).policy_result.compare_payload


def build_general_decision_artifact(
    *,
    now: datetime | None = None,
    use_bars: bool = False,
) -> dict[str, Any]:
    """Build a full read-only General Core artifact for demo inspection.

    The nested compare_payload remains the stable input for render_compare_text.
    This envelope carries the surrounding General path evidence without crossing
    confirmation, execution, or SDEP boundaries.
    """

    generated_at = now or datetime.now(timezone.utc).replace(microsecond=0)
    result = run_general_read_only_path(now=generated_at)
    compare_payload = result.policy_result.compare_payload
    selected_candidate_id = result.policy_result.checkpoint.selected_candidate_id
    return {
        "path_type": "read_only_general",
        "generated_by": "general_adapter",
        "decision_id": result.policy_result.checkpoint.decision_id,
        "trace_ref": result.policy_result.checkpoint.trace_ref,
        "selected_candidate_id": selected_candidate_id,
        "observations": [observation.to_payload() for observation in result.observations],
        "general_state_summary": _general_state_summary(result.state, compare_payload),
        "candidates": [candidate.to_payload() for candidate in result.candidates],
        "compare_payload": compare_payload,
        "rendered_text": render_compare_text(compare_payload, use_bars=use_bars),
    }


def decision_hub_general_guidance() -> DecisionGuidance:
    return DecisionGuidance(
        source_path="examples/decision_hub_demo/general_adapter.py",
        source_hash="decision_hub_general_adapter_v1",
        artifact_id="decision.decision_hub_demo.general_read_only",
        schema_version="0.1",
        artifact_version="0.1.0",
        status="example",
        primary_objective=PrimaryObjectiveGuidance(
            text="Select a bounded, reversible generic action for the current demo state.",
            direction="maximize",
        ),
        weights={
            "outcome_value": 0.40,
            "risk_reduction": 0.25,
            "reversibility": 0.20,
            "confidence_alignment": 0.15,
        },
        hard_constraints=[
            HardConstraintGuidance(
                id="no_declared_veto_violation",
                rule="do not select candidates blocked by declared availability constraints",
                severity="veto",
            )
        ],
    )


def _capability_observation(observation: Observation) -> GenericObservation:
    attrs = observation.attributes
    capability_id = str(attrs.get("capability_id") or f"capability.{observation.id}")
    availability = str(attrs.get("availability") or "unknown")
    return GenericObservation(
        observation_id=f"general.{observation.id}",
        kind=ObservationKind.CAPABILITY,
        source=_source(observation),
        subject=ObservationSubject(
            subject_id=capability_id,
            subject_type="general_capability",
            title="External capability available",
        ),
        summary="External capability is available for bounded delegation.",
        attributes={
            "capability_id": capability_id,
            "provider": str(attrs.get("executor") or observation.source or "unknown"),
            "scope": "general",
            "status": "available" if availability == "available" else availability,
            "requires_confirmation": attrs.get("requires_confirmation", True),
            "side_effects": ["external"] if attrs.get("supported_scopes") else [],
            "max_duration_seconds": int(attrs.get("default_time_budget_minutes") or 10) * 60,
        },
        confidence=_confidence(observation),
        refs=[observation.id],
    )


def _commitment_observation(observation: Observation) -> GenericObservation:
    attrs = observation.attributes
    commitment_id = str(attrs.get("commitment_id") or f"commitment.{observation.id}")
    return GenericObservation(
        observation_id=f"general.{observation.id}",
        kind=ObservationKind.COMMITMENT,
        source=_source(observation),
        subject=ObservationSubject(
            subject_id=commitment_id,
            subject_type="commitment",
            title=str(attrs.get("summary") or "Fixed commitment"),
        ),
        summary=str(attrs.get("summary") or "Fixed commitment"),
        attributes={
            "commitment_id": commitment_id,
            "title": str(attrs.get("summary") or "Fixed commitment"),
            "start_at": attrs.get("start_time"),
            "end_at": attrs.get("end_time"),
            "prep_start_at": attrs.get("prep_start_time"),
            "priority": attrs.get("priority_hint", "normal"),
            "fixed": str(attrs.get("flexibility_hint", "fixed")) == "fixed",
        },
        confidence=_confidence(observation),
        refs=[observation.id],
    )


def _work_item_observation(observation: Observation) -> GenericObservation:
    attrs = observation.attributes
    work_item_id = _work_item_id(observation)
    return GenericObservation(
        observation_id=f"general.{observation.id}",
        kind=ObservationKind.WORK_ITEM,
        source=_source(observation),
        subject=ObservationSubject(
            subject_id=work_item_id,
            subject_type="work_item",
            title=str(attrs.get("title") or "Untitled work item"),
        ),
        summary=str(attrs.get("title") or "Untitled work item"),
        attributes={
            "work_item_id": work_item_id,
            "title": str(attrs.get("title") or "Untitled work item"),
            "status": "open",
            "urgency": attrs.get("urgency_hint", "unknown"),
            "estimate_minutes": attrs.get("estimated_minutes_hint"),
            "owner": "user",
        },
        confidence=_confidence(observation),
        refs=[observation.id],
    )


def _intent_observation_for_work_item(observation: Observation) -> GenericObservation:
    attrs = observation.attributes
    work_item_id = _work_item_id(observation)
    return GenericObservation(
        observation_id=f"general.intent.{observation.id}",
        kind=ObservationKind.INTENT,
        source=_source(observation),
        subject=ObservationSubject(
            subject_id=f"intent.{stable_slug(work_item_id)}",
            subject_type="intent",
            title=f"Decide how to handle {attrs.get('title') or 'work item'}",
            refs=[work_item_id],
        ),
        summary=f"Decide how to handle work item: {attrs.get('title') or 'Untitled work item'}",
        attributes={
            "intent_id": f"intent.{stable_slug(work_item_id)}",
            "desired_outcome": "reduce work item risk while preserving active commitments",
            "urgency": attrs.get("urgency_hint", "unknown"),
            "target_refs": [work_item_id],
        },
        confidence=_confidence(observation),
        refs=[observation.id, work_item_id],
    )


def _derived_constraint_observations(
    observations: list[Observation],
    now: datetime,
) -> list[GenericObservation]:
    commitment = next(
        (item for item in observations if item.observation_type == "commitment_declared"),
        None,
    )
    work_item = next(
        (item for item in observations if item.observation_type == "work_item_opened"),
        None,
    )
    if not commitment or not work_item:
        return []

    prep_start = parse_time(commitment.attributes.get("prep_start_time"))
    estimate_minutes = _int_or_none(work_item.attributes.get("estimated_minutes_hint")) or 30
    if prep_start is None:
        return []

    available_minutes = max(0, int((prep_start - now).total_seconds() // 60))
    if available_minutes >= estimate_minutes:
        return []

    work_item_id = _work_item_id(work_item)
    constraint_id = f"constraint.time_window.{stable_slug(work_item_id)}"
    return [
        GenericObservation(
            observation_id=f"general.{constraint_id}",
            kind=ObservationKind.CONSTRAINT,
            source=ObservationSource(
                provider="decision_hub_demo",
                channel="derived_constraint",
                received_at=isoformat_utc(now),
            ),
            subject=ObservationSubject(
                subject_id=constraint_id,
                subject_type="time_window",
                title="Available window is shorter than estimated work",
                refs=[work_item_id],
            ),
            summary="Available window is shorter than estimated work.",
            attributes={
                "constraint_id": constraint_id,
                "constraint_kind": "time_window",
                "description": (
                    f"Available window ({available_minutes} minutes) is shorter "
                    f"than estimated work ({estimate_minutes} minutes)."
                ),
                "severity": "medium",
                "target_refs": [work_item_id],
                "available_minutes": available_minutes,
                "estimated_minutes": estimate_minutes,
            },
            confidence=ObservationConfidence(score=1.0, level="high"),
            refs=[commitment.id, work_item.id, work_item_id],
        )
    ]


def _unknown_observation(observation: Observation) -> GenericObservation:
    return GenericObservation(
        observation_id=f"general.{observation.id}",
        kind=ObservationKind.UNKNOWN,
        source=_source(observation),
        summary=f"Unsupported demo observation: {observation.observation_type}",
        attributes=dict(observation.attributes),
        confidence=_confidence(observation),
        refs=[observation.id],
    )


def _source(observation: Observation) -> ObservationSource:
    return ObservationSource(
        provider=observation.source or "unknown",
        channel=observation.observation_type,
        external_id=observation.id,
        received_at=isoformat_utc(observation.timestamp),
        metadata=dict(observation.metadata),
    )


def _confidence(observation: Observation) -> ObservationConfidence:
    raw = observation.metadata.get("confidence", observation.attributes.get("confidence", 1.0))
    try:
        score = float(raw)
    except (TypeError, ValueError):
        score = 1.0
    return ObservationConfidence(score=score, level="high" if score >= 0.75 else "low")


def _work_item_id(observation: Observation) -> str:
    attrs = observation.attributes
    repo = str(attrs.get("repo", "unknown"))
    item_id = str(attrs.get("item_id", attrs.get("id", "unknown")))
    return str(
        attrs.get("work_item_id")
        or f"workitem.{observation.source or 'unknown'}.{stable_slug(repo)}.{stable_slug(item_id)}"
    )


def _timestamp_slug(now: datetime) -> str:
    return isoformat_utc(now).replace("+00:00", "Z").replace(":", "").replace("-", "")


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _general_state_summary(
    state: GeneralDecisionState,
    compare_payload: dict[str, Any],
) -> dict[str, Any]:
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
        "decision_relevant_state_summary": compare_payload.get(
            "decision_relevant_state_summary",
            {},
        ),
    }
