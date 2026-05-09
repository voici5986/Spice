from __future__ import annotations

import json
import unittest

from spice.decision import (
    DecisionGuidance,
    HardConstraintGuidance,
    PrimaryObjectiveGuidance,
)
from spice.decision.compare import render_compare_text
from spice.decision.general import (
    GeneralDecisionState,
    GenericObservation,
    GenericPolicyAdapter,
    ObservationConfidence,
    ObservationKind,
    ObservationSource,
    ObservationSubject,
    generate_generic_candidates,
    reduce_generic_observations,
)


class GeneralDecisionFlowTests(unittest.TestCase):
    def test_observation_to_decision_card_flow_is_closed(self) -> None:
        observations = [
            _observation(
                observation_id="obs.intent.1",
                kind=ObservationKind.INTENT,
                subject=ObservationSubject(
                    subject_id="intent.1",
                    subject_type="intent",
                    refs=["item.1"],
                ),
                summary="Handle the incoming item.",
                attributes={
                    "desired_outcome": "item handled with bounded risk",
                    "urgency": "high",
                },
                confidence=ObservationConfidence(score=0.86, level="high"),
            ),
            _observation(
                observation_id="obs.item.1",
                kind=ObservationKind.WORK_ITEM,
                subject=ObservationSubject(
                    subject_id="item.1",
                    subject_type="work_item",
                    title="Review incoming item",
                ),
                summary="Incoming item needs a bounded first pass.",
                attributes={
                    "urgency": "high",
                    "estimate_minutes": "15",
                    "owner": "user",
                },
                confidence=ObservationConfidence(score=0.90, level="high"),
            ),
            _observation(
                observation_id="obs.capability.1",
                kind=ObservationKind.CAPABILITY,
                subject=ObservationSubject(
                    subject_id="capability.1",
                    subject_type="general_capability",
                ),
                summary="A local capability is available.",
                attributes={
                    "capability_id": "capability.local",
                    "provider": "local",
                    "scope": "general",
                    "requires_confirmation": "false",
                    "side_effects": ["read"],
                },
                confidence=ObservationConfidence(score=0.88, level="high"),
            ),
            _observation(
                observation_id="obs.constraint.1",
                kind=ObservationKind.CONSTRAINT,
                subject=ObservationSubject(
                    subject_id="constraint.1",
                    subject_type="bounded_attention",
                    refs=["item.1"],
                ),
                summary="Keep the first pass bounded.",
                attributes={
                    "constraint_id": "constraint.bounded_attention",
                    "constraint_kind": "attention",
                    "severity": "medium",
                },
                confidence=ObservationConfidence(score=0.80, level="high"),
            ),
        ]

        state = reduce_generic_observations(
            GeneralDecisionState(state_id="world.general"),
            observations,
        )
        candidates = generate_generic_candidates(state)
        result = GenericPolicyAdapter(_guidance()).evaluate(
            state,
            candidates=candidates,
            decision_id="decision.general.e2e",
            trace_ref="trace.general.e2e",
        )
        rendered = render_compare_text(result.compare_payload, use_bars=False)
        encoded = json.dumps(result.to_payload()).lower()

        self.assertEqual(len(state.observations), 4)
        self.assertEqual(state.intents[0].intent_id, "intent.1")
        self.assertEqual(state.work_items[0].work_item_id, "item.1")
        self.assertEqual(state.capabilities[0].capability_id, "capability.local")
        self.assertEqual(state.constraints[0].constraint_id, "constraint.bounded_attention")
        self.assertTrue(candidates)
        self.assertIn(
            result.checkpoint.selected_candidate_id,
            {candidate.candidate_id for candidate in candidates},
        )
        self.assertEqual(
            result.compare_payload["selected_recommendation"]["candidate_id"],
            result.checkpoint.selected_candidate_id,
        )
        self.assertIn("DECISION COMPARISON", rendered)
        self.assertIn("WHY NOT OTHERS", rendered)
        self.assertIn(result.checkpoint.selected_candidate_id, rendered)
        for banned in ("sdep", "hermes", "codex", "github", "whatsapp", "meeting", "flight"):
            self.assertNotIn(banned, encoded)

    def test_unknown_observation_can_only_be_recorded_without_external_boundary(self) -> None:
        observation = _observation(
            observation_id="obs.unknown.1",
            kind=ObservationKind.UNKNOWN,
            subject=ObservationSubject(
                subject_id="unknown.1",
                subject_type="unknown",
                title="Unclassified signal",
            ),
            summary="Unclassified signal arrived.",
            confidence=ObservationConfidence(score=0.95, level="high"),
        )
        state = reduce_generic_observations(
            GeneralDecisionState(state_id="world.general"),
            [observation],
        )

        candidates = generate_generic_candidates(state)
        action_types = {candidate.action_type for candidate in candidates}
        result = GenericPolicyAdapter(_guidance()).evaluate(
            state,
            candidates=candidates,
            decision_id="decision.general.unknown",
            trace_ref="trace.general.unknown",
        )
        selected = candidates[0]

        self.assertEqual(len(state.observations), 1)
        self.assertEqual(state.intents, [])
        self.assertEqual(state.work_items, [])
        self.assertEqual(state.risks, [])
        self.assertEqual(state.commitments, [])
        self.assertEqual(action_types, {"state.record"})
        self.assertNotIn("intent.execute", action_types)
        self.assertNotIn("item.triage", action_types)
        self.assertNotIn("capability.use", action_types)
        self.assertEqual(selected.side_effect_class, "none")
        self.assertFalse(selected.requires_confirmation)
        self.assertEqual(selected.execution_boundary.protocol, "")
        self.assertNotEqual(selected.execution_boundary.side_effect_class, "external")
        self.assertEqual(result.checkpoint.selected_candidate_id, selected.candidate_id)
        rendered = render_compare_text(result.compare_payload, use_bars=False)
        self.assertIn("DECISION COMPARISON", rendered)
        self.assertIn("state.record", rendered)


def _guidance() -> DecisionGuidance:
    return DecisionGuidance(
        source_path="test://general-decision.md",
        source_hash="guidance.test.general.e2e",
        artifact_id="decision.test.general",
        schema_version="0.1",
        artifact_version="0.1.0",
        status="test",
        primary_objective=PrimaryObjectiveGuidance(
            text="Select the best available generic action.",
            direction="maximize",
        ),
        weights={
            "outcome_value": 0.25,
            "risk_reduction": 0.15,
            "reversibility": 0.10,
            "confidence_alignment": 0.10,
            "urgency_alignment": 0.15,
            "effort_fit": 0.10,
            "impact_potential": 0.10,
            "preference_alignment": 0.05,
        },
        hard_constraints=[
            HardConstraintGuidance(
                id="no_declared_veto_violation",
                rule="do not select candidates blocked by declared availability constraints",
                severity="veto",
            )
        ],
    )


def _observation(
    *,
    observation_id: str,
    kind: ObservationKind | str,
    subject: ObservationSubject | None = None,
    summary: str = "",
    attributes: dict[str, object] | None = None,
    confidence: ObservationConfidence | None = None,
) -> GenericObservation:
    return GenericObservation(
        observation_id=observation_id,
        kind=kind,
        source=ObservationSource(provider="local"),
        subject=subject,
        summary=summary,
        attributes=dict(attributes or {}),
        confidence=confidence or ObservationConfidence(score=0.8, level="high"),
    )


if __name__ == "__main__":
    unittest.main()
