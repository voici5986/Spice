from __future__ import annotations

import json
import unittest

from spice.decision.general import (
    Approval,
    CandidateTraceRef,
    Capability,
    DecisionCheckpoint,
    DecisionTrace,
    GENERAL_STATE_KEY,
    GeneralDecisionState,
    GenericObservation,
    Intent,
    ObservationConfidence,
    ObservationEvidence,
    ObservationKind,
    ObservationSource,
    ObservationSubject,
    Signal,
    OpenLoop,
    OutcomeRecord,
    WorkItem,
    ensure_general_state,
    load_general_state,
    store_general_state,
)
from spice.protocols.observation import Observation
from spice.protocols.world_state import WorldState


class GeneralDecisionCoreTests(unittest.TestCase):
    def test_generic_observation_from_protocol_observation(self) -> None:
        observation = Observation(
            id="obs.manual.1",
            observation_type="work_item",
            source="manual",
            attributes={
                "subject_id": "work.item.1",
                "subject_type": "work_item",
                "title": "Review incoming item",
                "summary": "An incoming work item needs review.",
            },
        )

        generic = GenericObservation.from_protocol_observation(observation)

        self.assertEqual(generic.observation_id, "obs.manual.1")
        self.assertEqual(generic.kind, "work_item")
        self.assertEqual(generic.source.provider, "manual")
        self.assertIsNotNone(generic.subject)
        self.assertEqual(generic.subject.subject_id, "work.item.1")
        self.assertEqual(generic.summary, "An incoming work item needs review.")

    def test_general_state_round_trips_through_world_state(self) -> None:
        world = WorldState(id="world.general")
        ensure_general_state(world)

        state = GeneralDecisionState(
            state_id=world.id,
            signals=[
                Signal(
                    signal_id="signal.inbox.item.1",
                    source="inbox",
                    kind="work_item",
                    summary="A work item was received.",
                )
            ],
            observations=[
                GenericObservation(
                    observation_id="obs.inbox.item.1",
                    kind=ObservationKind.WORK_ITEM,
                    source=ObservationSource(provider="manual", channel="inbox"),
                    subject=ObservationSubject(
                        subject_id="work.item.1",
                        subject_type="work_item",
                        title="Review incoming item",
                    ),
                    evidence=[
                        ObservationEvidence(
                            evidence_id="evidence.inbox.item.1",
                            kind="message",
                            summary="Inbound message received.",
                        )
                    ],
                    confidence=ObservationConfidence(score=0.9, level="high"),
                )
            ],
            intents=[
                Intent(
                    intent_id="intent.handle_item",
                    summary="Decide how to handle the incoming item.",
                    target_refs=["work.item.1"],
                )
            ],
            work_items=[
                WorkItem(
                    work_item_id="work.item.1",
                    title="Review incoming item",
                    urgency="medium",
                )
            ],
            capabilities=[
                Capability(
                    capability_id="cap.executor.review",
                    provider="local_executor",
                    scope="review",
                    side_effects=["file_read"],
                )
            ],
            trace_refs=["trace.demo"],
        )

        store_general_state(world, state)
        loaded = load_general_state(world)

        self.assertIn(GENERAL_STATE_KEY, world.domain_state)
        self.assertEqual(loaded.state_id, "world.general")
        self.assertEqual(loaded.signals[0].signal_id, "signal.inbox.item.1")
        self.assertEqual(loaded.observations[0].source.provider, "manual")
        self.assertEqual(loaded.observations[0].subject.subject_id, "work.item.1")
        self.assertEqual(loaded.capabilities[0].provider, "local_executor")
        self.assertEqual(loaded.trace_refs, ["trace.demo"])

    def test_store_general_state_preserves_other_domain_state_namespaces(self) -> None:
        world = WorldState(
            id="world.general",
            domain_state={"other_namespace": {"kept": True}},
        )

        store_general_state(world, GeneralDecisionState(state_id=world.id))

        self.assertEqual(world.domain_state["other_namespace"], {"kept": True})
        self.assertIn(GENERAL_STATE_KEY, world.domain_state)

    def test_ensure_general_state_repairs_invalid_domain_state(self) -> None:
        for invalid in (None, [], "invalid"):
            world = WorldState(id="world.general")
            world.domain_state = invalid  # type: ignore[assignment]

            general = ensure_general_state(world)

            self.assertIsInstance(world.domain_state, dict)
            self.assertIsInstance(general, dict)
            self.assertIn("schema_version", general)

    def test_ensure_general_state_repairs_invalid_general_namespace(self) -> None:
        for invalid in (None, [], "invalid"):
            world = WorldState(
                id="world.general",
                domain_state={GENERAL_STATE_KEY: invalid, "other": {"kept": True}},
            )

            general = ensure_general_state(world)

            self.assertIsInstance(general, dict)
            self.assertEqual(world.domain_state["other"], {"kept": True})
            self.assertEqual(world.domain_state[GENERAL_STATE_KEY], general)

    def test_load_general_state_ignores_unknown_snapshot_fields(self) -> None:
        world = WorldState(
            id="world.general",
            domain_state={
                GENERAL_STATE_KEY: {
                    "signals": [
                        {
                            "signal_id": "signal.1",
                            "source": "manual",
                            "kind": "intent",
                            "summary": "A signal.",
                            "future_field": "ignored",
                        }
                    ],
                    "outcomes": [
                        {
                            "outcome_id": "outcome.1",
                            "decision_id": "decision.1",
                            "trace_ref": "trace.1",
                            "candidate_id": "candidate.1",
                            "execution_ref": "execution.1",
                            "future_field": "ignored",
                        }
                    ],
                }
            },
        )

        loaded = load_general_state(world)

        self.assertEqual(loaded.signals[0].signal_id, "signal.1")
        self.assertEqual(loaded.outcomes[0].trace_ref, "trace.1")
        self.assertEqual(loaded.outcomes[0].candidate_id, "candidate.1")

    def test_decision_trace_refs_are_stable(self) -> None:
        trace = DecisionTrace(
            trace_ref="trace.abc",
            decision_id="decision.abc",
            state_ref="world.general",
            profile_ref="profile.default",
            observation_refs=["obs.manual.item.1"],
            candidate_refs=[
                CandidateTraceRef(
                    candidate_id="candidate.execute_intent",
                    action_type="execute_intent",
                    status="selected",
                )
            ],
            selected_candidate_id="candidate.execute_intent",
            execution_ref="exec.abc",
            outcome_refs=["outcome.abc"],
        )

        refs = trace.refs()
        payload = refs.to_payload()

        self.assertEqual(payload["decision_id"], "decision.abc")
        self.assertEqual(payload["trace_ref"], "trace.abc")
        self.assertEqual(payload["state_ref"], "world.general")
        self.assertEqual(payload["profile_ref"], "profile.default")
        self.assertEqual(payload["candidate_refs"], ["candidate.execute_intent"])
        self.assertEqual(payload["execution_ref"], "exec.abc")

    def test_decision_checkpoint_keeps_approval_before_execution_boundary(self) -> None:
        checkpoint = DecisionCheckpoint(
            decision_id="decision.abc",
            trace_ref="trace.abc",
            state_ref="world.general",
            profile_ref="profile.default",
            selected_candidate_id="candidate.execute_intent",
            recommendation="Execute the selected intent after approval.",
            candidate_refs=[
                CandidateTraceRef(
                    candidate_id="candidate.execute_intent",
                    action_type="execute_intent",
                    status="selected",
                )
            ],
            approval=Approval(
                approval_id="approval.abc",
                decision_id="decision.abc",
                candidate_id="candidate.execute_intent",
                status="pending",
                execution_allowed=False,
            ),
            execution_boundary={"protocol": "sdep", "request_ref": "execute.request.pending"},
        )

        payload = checkpoint.to_payload()

        self.assertEqual(payload["approval"]["status"], "pending")
        self.assertFalse(payload["approval"]["execution_allowed"])
        self.assertEqual(payload["execution_boundary"]["protocol"], "sdep")
        self.assertEqual(payload["selected_candidate_id"], "candidate.execute_intent")

    def test_outcome_record_supports_decision_trace_candidate_attribution(self) -> None:
        outcome = OutcomeRecord(
            outcome_id="outcome.abc",
            decision_id="decision.abc",
            trace_ref="trace.abc",
            candidate_id="candidate.abc",
            execution_ref="execution.abc",
            protocol_status="success",
            task_status="failed",
        )

        payload = outcome.to_payload()

        self.assertEqual(payload["decision_id"], "decision.abc")
        self.assertEqual(payload["trace_ref"], "trace.abc")
        self.assertEqual(payload["candidate_id"], "candidate.abc")
        self.assertEqual(payload["execution_ref"], "execution.abc")
        self.assertEqual(payload["protocol_status"], "success")
        self.assertEqual(payload["task_status"], "failed")

    def test_approval_statuses_can_express_minimal_flow(self) -> None:
        statuses = ["pending", "approved", "rejected", "needs_details"]

        approvals = [
            Approval(
                approval_id=f"approval.{status}",
                decision_id="decision.abc",
                status=status,
                execution_allowed=status == "approved",
            )
            for status in statuses
        ]

        self.assertEqual([item.status for item in approvals], statuses)
        self.assertFalse(approvals[0].execution_allowed)
        self.assertTrue(approvals[1].execution_allowed)

    def test_open_loop_can_express_minimal_lifecycle(self) -> None:
        open_loop = OpenLoop(
            open_loop_id="loop.abc",
            summary="Follow up on a pending item.",
            status="open",
        )
        updated_loop = OpenLoop(
            open_loop_id=open_loop.open_loop_id,
            summary=open_loop.summary,
            status="updated",
            metadata={"last_update": "new evidence received"},
        )
        closed_loop = OpenLoop(
            open_loop_id=open_loop.open_loop_id,
            summary=open_loop.summary,
            status="closed",
            metadata={"closed_reason": "resolved"},
        )

        self.assertEqual(open_loop.status, "open")
        self.assertEqual(updated_loop.metadata["last_update"], "new evidence received")
        self.assertEqual(closed_loop.status, "closed")

    def test_general_state_json_round_trip(self) -> None:
        world = WorldState(id="world.general")
        original = GeneralDecisionState(
            state_id=world.id,
            signals=[
                Signal(
                    signal_id="signal.1",
                    source="manual",
                    kind="intent",
                    summary="A user intent was received.",
                )
            ],
            outcomes=[
                OutcomeRecord(
                    outcome_id="outcome.1",
                    decision_id="decision.1",
                    trace_ref="trace.1",
                    candidate_id="candidate.1",
                    execution_ref="execution.1",
                )
            ],
        )

        store_general_state(world, original)
        encoded = json.dumps(world.domain_state)
        decoded = json.loads(encoded)
        restored = WorldState(id=world.id, domain_state=decoded)
        loaded = load_general_state(restored)

        self.assertEqual(loaded.signals[0].signal_id, "signal.1")
        self.assertEqual(loaded.outcomes[0].decision_id, "decision.1")
        self.assertEqual(loaded.outcomes[0].trace_ref, "trace.1")

    def test_observation_kind_serializes_as_stable_string(self) -> None:
        observation = GenericObservation(
            observation_id="obs.1",
            kind=ObservationKind.INTENT,
            source=ObservationSource(provider="manual"),
        )

        payload = observation.to_payload()

        self.assertEqual(payload["kind"], "intent")


if __name__ == "__main__":
    unittest.main()
