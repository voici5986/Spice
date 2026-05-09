from __future__ import annotations

import json
import unittest

from spice.decision.general import (
    Capability,
    Constraint,
    GENERIC_ACTION_TYPES,
    GeneralDecisionState,
    GenericCandidate,
    GenericObservation,
    Intent,
    ObservationConfidence,
    ObservationKind,
    ObservationSource,
    ObservationSubject,
    Signal,
    WorkItem,
    generate_generic_candidates,
)


class GenericCandidateLayerTests(unittest.TestCase):
    def test_candidate_schema_round_trips_with_nested_fields(self) -> None:
        candidate = GenericCandidate.from_payload(
            {
                "candidate_id": "candidate.intent_execute.intent_1",
                "action_type": "intent.execute",
                "intent": "Act on intent.",
                "candidate_kind": "execution_handoff",
                "target_refs": ["intent.1"],
                "execution_intent": {
                    "intent_class": "execution_requested",
                    "requested": True,
                    "handoff_task": "Act on intent.",
                    "required_permission_hint": "workspace_write",
                    "side_effect_class": "external_effect",
                },
                "estimated_cost": {"time_minutes": 3, "future": "ignored"},
                "risk_profile": {"level": "low"},
                "expected_state_delta": {"updates_refs": ["intent.1"]},
                "execution_boundary": {
                    "mode": "execution_intent",
                    "protocol": "custom",
                },
                "future_field": "ignored",
            }
        )

        encoded = json.dumps(candidate.to_payload())
        decoded = GenericCandidate.from_payload(json.loads(encoded))

        self.assertEqual(decoded.candidate_id, "candidate.intent_execute.intent_1")
        self.assertEqual(decoded.candidate_kind, "execution_handoff")
        self.assertEqual(decoded.execution_intent.intent_class, "execution_requested")
        self.assertTrue(decoded.execution_intent.requested)
        self.assertEqual(decoded.execution_intent.handoff_task, "Act on intent.")
        self.assertEqual(decoded.execution_intent.required_permission_hint, "workspace_write")
        self.assertEqual(decoded.execution_intent.side_effect_class, "external_effect")
        self.assertEqual(decoded.estimated_cost.time_minutes, 3)
        self.assertEqual(decoded.risk_profile.level, "low")
        self.assertEqual(decoded.execution_boundary.protocol, "custom")

    def test_candidate_schema_reads_kind_and_execution_intent_from_metadata(self) -> None:
        candidate = GenericCandidate.from_payload(
            {
                "candidate_id": "candidate.llm.decision_1",
                "action_type": "item.triage",
                "intent": "Compare release priorities.",
                "metadata": {
                    "candidate_kind": "decision",
                    "execution_intent": {
                        "needs_execution": False,
                        "reason": "Advisory decision only.",
                        "side_effect": "read_only",
                    },
                },
            }
        )

        self.assertEqual(candidate.candidate_kind, "decision")
        self.assertFalse(candidate.execution_intent.requested)
        self.assertEqual(candidate.execution_intent.intent_class, "advisory")
        self.assertEqual(candidate.execution_intent.reason, "Advisory decision only.")
        self.assertEqual(candidate.execution_intent.side_effect_class, "read_only")

    def test_generates_candidates_without_selecting_or_scoring(self) -> None:
        state = GeneralDecisionState(
            state_id="world.general",
            intents=[
                Intent(
                    intent_id="intent.1",
                    summary="Handle a new user request.",
                    target_refs=["target.1"],
                )
            ],
            work_items=[
                WorkItem(
                    work_item_id="work.1",
                    title="Handle incoming item",
                    estimate_minutes=90,
                )
            ],
        )

        candidates = generate_generic_candidates(state)
        payloads = [candidate.to_payload() for candidate in candidates]
        action_types = {candidate.action_type for candidate in candidates}

        self.assertIn("intent.execute", action_types)
        self.assertIn("context.prepare", action_types)
        self.assertIn("time.defer", action_types)
        self.assertIn("item.triage", action_types)
        self.assertIn("task.split", action_types)
        for payload in payloads:
            self.assertNotIn("selected", payload)
            self.assertNotIn("score_total", payload)
            self.assertNotIn("score_breakdown", payload)
            self.assertEqual(payload["candidate_kind"], "runtime_action")
            self.assertEqual(payload["metadata"]["candidate_kind"], "runtime_action")
            self.assertEqual(payload["metadata"]["candidate_source"], "rule")

    def test_delegate_candidate_uses_capability_but_does_not_pick_executor(self) -> None:
        state = GeneralDecisionState(
            state_id="world.general",
            intents=[
                Intent(
                    intent_id="intent.1",
                    summary="Act on a request.",
                    target_refs=["target.1"],
                )
            ],
            capabilities=[
                Capability(
                    capability_id="cap.review",
                    provider="local_executor",
                    scope="review",
                    requires_confirmation=True,
                    side_effects=["file_write"],
                )
            ],
        )

        candidates = generate_generic_candidates(state)
        capability_use = _candidate_by_action(candidates, "capability.use")
        permission = _candidate_by_action(candidates, "approval.request")

        self.assertEqual(capability_use.required_capability, "cap.review")
        self.assertEqual(capability_use.execution_boundary.mode, "capability")
        self.assertEqual(capability_use.execution_boundary.protocol, "")
        self.assertTrue(capability_use.execution_intent.requested)
        self.assertEqual(capability_use.execution_intent.intent_class, "execution_requested")
        self.assertIn("Delegate intent", capability_use.execution_intent.handoff_task)
        self.assertTrue(capability_use.requires_confirmation)
        self.assertEqual(capability_use.availability_status, "needs_confirmation")
        self.assertEqual(capability_use.side_effect_class, "external")
        self.assertEqual(permission.action_type, "approval.request")

    def test_constraints_can_mark_candidate_blocked_without_deciding(self) -> None:
        state = GeneralDecisionState(
            state_id="world.general",
            intents=[
                Intent(
                    intent_id="intent.1",
                    summary="Act on a request.",
                    target_refs=["target.1"],
                )
            ],
            constraints=[
                Constraint(
                    constraint_id="constraint.requires_permission",
                    kind="approval",
                    description="Approval is required before side effects.",
                    severity="veto",
                    applies_to_refs=["target.1"],
                )
            ],
        )

        candidates = generate_generic_candidates(state)
        execute = _candidate_by_action(candidates, "intent.execute")

        self.assertEqual(execute.availability_status, "blocked")
        self.assertEqual(
            execute.constraints_triggered[0]["constraint_id"],
            "constraint.requires_permission",
        )
        self.assertEqual(
            execute.why_blocked,
            ["Approval is required before side effects."],
        )
        self.assertFalse(any("selected" in item.to_payload() for item in candidates))

    def test_uncertain_observation_generates_clarification_and_observe_more(self) -> None:
        state = GeneralDecisionState(
            state_id="world.general",
            observations=[
                GenericObservation(
                    observation_id="obs.1",
                    kind=ObservationKind.INTENT,
                    source=ObservationSource(provider="manual"),
                    subject=ObservationSubject(
                        subject_id="subject.1",
                        subject_type="intent",
                    ),
                    confidence=ObservationConfidence(
                        level="low",
                        missing_fields=["deadline"],
                        uncertain_fields=["priority"],
                    ),
                )
            ],
        )

        candidates = generate_generic_candidates(state)
        action_types = {candidate.action_type for candidate in candidates}
        clarification = _candidate_by_action(candidates, "user.clarify")

        self.assertIn("user.clarify", action_types)
        self.assertIn("state.observe_more", action_types)
        self.assertIn("state.record", action_types)
        self.assertEqual(clarification.target_refs, ["subject.1", "obs.1"])
        self.assertFalse(clarification.requires_confirmation)

    def test_action_type_contract_is_stable(self) -> None:
        self.assertEqual(
            set(GENERIC_ACTION_TYPES),
            {
                "intent.execute",
                "capability.use",
                "item.triage",
                "context.prepare",
                "state.observe_more",
                "artifact.draft",
                "approval.request",
                "user.clarify",
                "time.defer",
                "state.record",
                "item.ignore",
                "task.split",
            },
        )

    def test_empty_state_generates_no_candidates(self) -> None:
        self.assertEqual(
            generate_generic_candidates(GeneralDecisionState(state_id="world.general")),
            [],
        )

    def test_candidate_ids_are_stable_for_same_state(self) -> None:
        state = GeneralDecisionState(
            state_id="world.general",
            intents=[
                Intent(
                    intent_id="intent.1",
                    summary="Handle a request.",
                    target_refs=["target.1"],
                )
            ],
            capabilities=[
                Capability(
                    capability_id="cap.prepare",
                    provider="local_provider",
                    scope="prepare",
                )
            ],
        )

        first = [candidate.candidate_id for candidate in generate_generic_candidates(state)]
        second = [candidate.candidate_id for candidate in generate_generic_candidates(state)]

        self.assertEqual(first, second)

    def test_blocked_candidate_remains_visible(self) -> None:
        state = GeneralDecisionState(
            state_id="world.general",
            intents=[
                Intent(
                    intent_id="intent.1",
                    summary="Handle a request.",
                    target_refs=["target.1"],
                )
            ],
            constraints=[
                Constraint(
                    constraint_id="constraint.block",
                    kind="safety",
                    description="Candidate is blocked by a constraint.",
                    severity="veto",
                    applies_to_refs=["target.1"],
                )
            ],
        )

        candidates = generate_generic_candidates(state)

        self.assertTrue(
            any(candidate.availability_status == "blocked" for candidate in candidates)
        )
        self.assertIn("intent.execute", {candidate.action_type for candidate in candidates})

    def test_action_types_do_not_contain_demo_or_executor_terms(self) -> None:
        banned_terms = (
            "executor",
            "github",
            "pr",
            "codex",
            "meeting",
            "flight",
            "delegate_to_executor",
        )

        for action_type in GENERIC_ACTION_TYPES:
            segments = action_type.split(".")
            for term in banned_terms:
                self.assertNotIn(term, segments)

    def test_generated_execution_boundaries_are_protocol_neutral(self) -> None:
        state = GeneralDecisionState(
            state_id="world.general",
            intents=[
                Intent(
                    intent_id="intent.1",
                    summary="Handle a request.",
                    target_refs=["target.1"],
                )
            ],
            capabilities=[
                Capability(
                    capability_id="cap.prepare",
                    provider="local_provider",
                    scope="prepare",
                )
            ],
            work_items=[WorkItem(work_item_id="work.1", title="Handle item")],
        )

        candidates = generate_generic_candidates(state)

        self.assertTrue(candidates)
        self.assertTrue(
            all(candidate.execution_boundary.protocol != "sdep" for candidate in candidates)
        )

    def test_expected_state_delta_does_not_mutate_state(self) -> None:
        state = GeneralDecisionState(
            state_id="world.general",
            work_items=[WorkItem(work_item_id="work.1", title="Handle item")],
        )
        before = state.to_payload()

        candidates = generate_generic_candidates(state)
        deltas = [candidate.expected_state_delta.to_payload() for candidate in candidates]

        self.assertTrue(deltas)
        self.assertEqual(state.to_payload(), before)

    def test_duplicate_candidate_ids_are_deduped(self) -> None:
        state = GeneralDecisionState(
            state_id="world.general",
            signals=[
                Signal(
                    signal_id="signal.1",
                    source="manual",
                    kind="signal",
                    summary="First signal.",
                ),
                Signal(
                    signal_id="signal.1",
                    source="manual",
                    kind="signal",
                    summary="Duplicate signal.",
                ),
            ],
        )

        candidates = generate_generic_candidates(state)
        candidate_ids = [candidate.candidate_id for candidate in candidates]

        self.assertEqual(len(candidate_ids), len(set(candidate_ids)))


def _candidate_by_action(
    candidates: list[GenericCandidate],
    action_type: str,
) -> GenericCandidate:
    for candidate in candidates:
        if candidate.action_type == action_type:
            return candidate
    raise AssertionError(f"candidate action not found: {action_type}")


if __name__ == "__main__":
    unittest.main()
