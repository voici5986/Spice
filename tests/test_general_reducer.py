from __future__ import annotations

import json
import unittest

from spice.decision.general import (
    GeneralDecisionState,
    GenericObservation,
    ObservationConfidence,
    ObservationEvidence,
    ObservationKind,
    ObservationSource,
    ObservationSubject,
    apply_generic_observation,
    reduce_generic_observations,
    store_general_state,
    load_general_state,
)
from spice.protocols.world_state import WorldState


class GeneralReducerTests(unittest.TestCase):
    def test_reduces_work_item_observation_without_mutating_original_state(self) -> None:
        original = GeneralDecisionState(state_id="world.general")
        observation = _observation(
            observation_id="obs.work.1",
            kind=ObservationKind.WORK_ITEM,
            subject=ObservationSubject(
                subject_id="work.1",
                subject_type="work_item",
                title="Review incoming item",
            ),
            summary="Incoming item needs review.",
            attributes={
                "urgency": "high",
                "estimate_minutes": "12",
                "owner": "user",
            },
        )

        reduced = apply_generic_observation(original, observation)

        self.assertEqual(original.observations, [])
        self.assertEqual(original.work_items, [])
        self.assertEqual(reduced.observations[0].observation_id, "obs.work.1")
        self.assertEqual(reduced.work_items[0].work_item_id, "work.1")
        self.assertEqual(reduced.work_items[0].title, "Review incoming item")
        self.assertEqual(reduced.work_items[0].urgency, "high")
        self.assertEqual(reduced.work_items[0].estimate_minutes, 12)
        self.assertEqual(
            reduced.work_items[0].metadata["observation_id"],
            "obs.work.1",
        )

    def test_reduces_common_observation_kinds_to_general_state_objects(self) -> None:
        state = reduce_generic_observations(
            GeneralDecisionState(state_id="world.general"),
            [
                _observation(
                    "obs.signal.1",
                    ObservationKind.SIGNAL,
                    summary="A provider event arrived.",
                    attributes={"signal_id": "signal.1", "kind": "provider_event"},
                ),
                _observation(
                    "obs.intent.1",
                    ObservationKind.INTENT,
                    subject=ObservationSubject(
                        subject_id="intent.1",
                        subject_type="intent",
                        refs=["target.1"],
                    ),
                    summary="Handle a user request.",
                    attributes={"desired_outcome": "request handled", "urgency": "medium"},
                    confidence=ObservationConfidence(score=0.8, level="high"),
                ),
                _observation(
                    "obs.commitment.1",
                    ObservationKind.COMMITMENT,
                    subject=ObservationSubject(
                        subject_id="commitment.1",
                        subject_type="commitment",
                        title="Fixed appointment",
                    ),
                    attributes={
                        "start_at": "2026-04-27T10:00:00Z",
                        "end_at": "2026-04-27T11:00:00Z",
                        "fixed": "true",
                    },
                ),
                _observation(
                    "obs.capability.1",
                    ObservationKind.CAPABILITY,
                    subject=ObservationSubject(
                        subject_id="capability.1",
                        subject_type="general_execution",
                    ),
                    attributes={
                        "provider": "local",
                        "scope": "general",
                        "requires_confirmation": "false",
                        "side_effects": ["external"],
                    },
                ),
                _observation(
                    "obs.constraint.1",
                    ObservationKind.CONSTRAINT,
                    subject=ObservationSubject(
                        subject_id="constraint.1",
                        subject_type="approval",
                        refs=["target.1"],
                    ),
                    summary="Approval is required.",
                    attributes={"severity": "veto"},
                ),
                _observation(
                    "obs.risk.1",
                    ObservationKind.RISK,
                    subject=ObservationSubject(
                        subject_id="risk.1",
                        subject_type="timing",
                        refs=["target.1"],
                    ),
                    summary="Timing risk exists.",
                    attributes={"level": "medium"},
                ),
                _observation(
                    "obs.outcome.1",
                    ObservationKind.OUTCOME,
                    summary="Execution returned a partial result.",
                    attributes={
                        "outcome_id": "outcome.1",
                        "decision_id": "decision.1",
                        "trace_ref": "trace.1",
                        "candidate_id": "candidate.1",
                        "execution_ref": "execution.1",
                        "protocol_status": "success",
                        "task_status": "partial",
                        "state_delta": {"updated": ["target.1"]},
                    },
                    evidence=[
                        ObservationEvidence(
                            evidence_id="evidence.1",
                            kind="message",
                        )
                    ],
                ),
            ],
        )

        self.assertEqual(state.signals[0].signal_id, "signal.1")
        self.assertEqual(state.intents[0].intent_id, "intent.1")
        self.assertEqual(state.intents[0].confidence, 0.8)
        self.assertEqual(state.commitments[0].commitment_id, "commitment.1")
        self.assertTrue(state.commitments[0].fixed)
        self.assertEqual(state.capabilities[0].capability_id, "capability.1")
        self.assertFalse(state.capabilities[0].requires_confirmation)
        self.assertEqual(state.constraints[0].constraint_id, "constraint.1")
        self.assertEqual(state.constraints[0].applies_to_refs, ["target.1"])
        self.assertEqual(state.risks[0].risk_id, "risk.1")
        self.assertEqual(state.risks[0].level, "medium")
        self.assertEqual(state.outcomes[0].decision_id, "decision.1")
        self.assertEqual(state.outcomes[0].trace_ref, "trace.1")
        self.assertEqual(state.outcomes[0].evidence_refs, ["evidence.1"])

    def test_reduces_resource_and_open_loop_string_kinds(self) -> None:
        state = reduce_generic_observations(
            GeneralDecisionState(state_id="world.general"),
            [
                _observation(
                    "obs.resource.1",
                    "resource",
                    subject=ObservationSubject(
                        subject_id="resource.1",
                        subject_type="time_budget",
                    ),
                    attributes={"capacity": {"minutes": 30}, "constraints": ["fixed"]},
                ),
                _observation(
                    "obs.loop.1",
                    "open_loop",
                    subject=ObservationSubject(
                        subject_id="loop.1",
                        subject_type="follow_up",
                        refs=["target.1"],
                    ),
                    summary="Follow up later.",
                    attributes={"owner": "user", "due_at": "2026-04-28"},
                ),
            ],
        )

        self.assertEqual(state.resources[0].resource_id, "resource.1")
        self.assertEqual(state.resources[0].capacity, {"minutes": 30})
        self.assertEqual(state.open_loops[0].open_loop_id, "loop.1")
        self.assertEqual(state.open_loops[0].target_refs, ["target.1"])

    def test_unknown_observation_kind_only_records_observation(self) -> None:
        state = apply_generic_observation(
            GeneralDecisionState(state_id="world.general"),
            _observation(
                observation_id="obs.unknown.1",
                kind=ObservationKind.UNKNOWN,
                summary="Unknown signal.",
            ),
        )

        self.assertEqual(len(state.observations), 1)
        self.assertEqual(state.signals, [])
        self.assertEqual(state.intents, [])
        self.assertEqual(state.work_items, [])
        self.assertEqual(state.risks, [])

    def test_reducer_upserts_by_stable_object_id(self) -> None:
        initial = _observation(
            observation_id="obs.work.initial",
            kind=ObservationKind.WORK_ITEM,
            subject=ObservationSubject(
                subject_id="work.1",
                subject_type="work_item",
                title="Initial title",
            ),
            attributes={"urgency": "low"},
        )
        updated = _observation(
            observation_id="obs.work.updated",
            kind=ObservationKind.WORK_ITEM,
            subject=ObservationSubject(
                subject_id="work.1",
                subject_type="work_item",
                title="Updated title",
            ),
            attributes={"urgency": "high"},
        )

        state = reduce_generic_observations(
            GeneralDecisionState(state_id="world.general"),
            [initial, updated],
        )

        self.assertEqual(len(state.work_items), 1)
        self.assertEqual(state.work_items[0].title, "Updated title")
        self.assertEqual(state.work_items[0].urgency, "high")
        self.assertEqual(len(state.observations), 2)

    def test_reduced_state_round_trips_through_world_state_json(self) -> None:
        state = apply_generic_observation(
            GeneralDecisionState(state_id="world.general"),
            _observation(
                observation_id="obs.intent.1",
                kind=ObservationKind.INTENT,
                subject=ObservationSubject(
                    subject_id="intent.1",
                    subject_type="intent",
                ),
                summary="Handle a request.",
            ),
        )
        world = WorldState(id="world.general")

        store_general_state(world, state)
        restored = WorldState(id=world.id, domain_state=json.loads(json.dumps(world.domain_state)))
        loaded = load_general_state(restored)

        self.assertEqual(loaded.observations[0].observation_id, "obs.intent.1")
        self.assertEqual(loaded.intents[0].intent_id, "intent.1")


def _observation(
    observation_id: str,
    kind: ObservationKind | str,
    *,
    subject: ObservationSubject | None = None,
    summary: str = "",
    attributes: dict[str, object] | None = None,
    confidence: ObservationConfidence | None = None,
    evidence: list[ObservationEvidence] | None = None,
) -> GenericObservation:
    return GenericObservation(
        observation_id=observation_id,
        kind=kind,
        source=ObservationSource(
            provider="test_provider",
            channel="test",
            external_id=f"external.{observation_id}",
            received_at="2026-04-27T00:00:00Z",
        ),
        subject=subject,
        summary=summary,
        attributes=dict(attributes or {}),
        confidence=confidence or ObservationConfidence(level="high"),
        evidence=list(evidence or []),
    )


if __name__ == "__main__":
    unittest.main()
