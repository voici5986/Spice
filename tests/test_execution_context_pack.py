from __future__ import annotations

import copy
import json
import unittest

from spice.decision.general import (
    Capability,
    Commitment,
    Constraint,
    GeneralDecisionState,
    GenericObservation,
    Intent,
    ObservationKind,
    ObservationSource,
    ObservationSubject,
    OutcomeRecord,
    Risk,
    Signal,
    WorkItem,
)
from spice.decision.general.candidates import (
    EstimatedCost,
    ExecutionBoundary,
    ExpectedStateDelta,
    GenericCandidate,
    RiskProfile,
)
from spice.executors import (
    ExecutionContextPack,
    ResolvedSkill,
    build_execution_context_pack,
)


class ExecutionContextPackTests(unittest.TestCase):
    def test_builds_compact_context_pack_for_resolved_candidate(self) -> None:
        state = _state()
        candidate = _candidate()
        resolved = _resolved_skill()

        pack = build_execution_context_pack(
            state=state,
            candidate=candidate,
            resolved_skill=resolved,
            decision_id="decision.1",
            trace_ref="trace.1",
            approval_id="approval.1",
            execution_id="execution.1",
            request_id="request.1",
        )

        self.assertTrue(pack.context_pack_id.startswith("context_pack."))
        self.assertEqual(pack.state_ref, "world.general")
        self.assertEqual(pack.decision_id, "decision.1")
        self.assertEqual(pack.trace_ref, "trace.1")
        self.assertEqual(pack.candidate_id, "candidate.item.triage.work.1")
        self.assertEqual(pack.skill_id, "work_item.triage.codex")
        self.assertEqual(pack.executor_id, "codex")
        self.assertEqual(pack.action_type, "item.triage")
        self.assertEqual(pack.target_refs, ["work.1"])
        self.assertEqual(pack.traceability["execution_id"], "execution.1")
        self.assertEqual(pack.traceability["request_id"], "request.1")
        self.assertTrue(pack.metadata["context_is_compact"])
        self.assertFalse(pack.metadata["external_memory_loaded"])
        self.assertTrue(pack.task)
        self.assertTrue(pack.why_now)
        self.assertTrue(pack.do_not)
        self.assertTrue(pack.expected_output)
        self.assertEqual(pack.return_schema["type"], "triage_report.v1")
        self.assertEqual(pack.instructions, ["Do not edit files.", "Return a concise triage summary."])

        relevant = pack.relevant_state
        self.assertEqual([item["work_item_id"] for item in relevant["work_items"]], ["work.1"])
        self.assertEqual([item["intent_id"] for item in relevant["intents"]], ["intent.1"])
        self.assertEqual([item["capability_id"] for item in relevant["capabilities"]], ["work_item_triage"])
        self.assertEqual([item["constraint_id"] for item in relevant["constraints"]], ["constraint.time"])
        self.assertEqual([item["risk_id"] for item in relevant["risks"]], ["risk.stale_work"])
        self.assertEqual([item["outcome_id"] for item in relevant["outcomes"]], ["outcome.1"])
        self.assertEqual(len(relevant["commitments"]), 1)
        self.assertEqual(relevant["work_items"][0]["title"], "Review incoming work")

        payload = json.loads(json.dumps(pack.to_payload()))
        restored = ExecutionContextPack.from_payload(payload)
        self.assertEqual(restored.context_pack_id, pack.context_pack_id)
        self.assertEqual(restored.resolved_skill["skill_id"], "work_item.triage.codex")
        self.assertEqual(restored.return_schema["type"], "triage_report.v1")
        self.assertEqual(restored.do_not, pack.do_not)

    def test_context_pack_id_is_deterministic(self) -> None:
        state = _state()
        candidate = _candidate()
        resolved = _resolved_skill()

        first = build_execution_context_pack(
            state=state,
            candidate=candidate,
            resolved_skill=resolved,
            decision_id="decision.1",
            trace_ref="trace.1",
            approval_id="approval.1",
            execution_id="execution.1",
            request_id="request.1",
        )
        second = build_execution_context_pack(
            state=state,
            candidate=candidate,
            resolved_skill=resolved,
            decision_id="decision.1",
            trace_ref="trace.1",
            approval_id="approval.1",
            execution_id="execution.1",
            request_id="request.1",
        )
        changed = build_execution_context_pack(
            state=state,
            candidate=candidate,
            resolved_skill=ResolvedSkill(
                executor_id="codex",
                skill_id="work_item.triage.other",
                action_type="item.triage",
                capability_id="work_item_triage",
                side_effect_class="read_only",
                requires_confirmation=False,
                metadata={"candidate_id": candidate.candidate_id},
            ),
            decision_id="decision.1",
            trace_ref="trace.1",
            approval_id="approval.1",
            execution_id="execution.1",
            request_id="request.1",
        )

        self.assertEqual(first.context_pack_id, second.context_pack_id)
        self.assertNotEqual(first.context_pack_id, changed.context_pack_id)

    def test_context_pack_id_changes_with_handoff_attribution(self) -> None:
        state = _state()
        candidate = _candidate()
        resolved = _resolved_skill()
        base = build_execution_context_pack(
            state=state,
            candidate=candidate,
            resolved_skill=resolved,
            decision_id="decision.1",
            trace_ref="trace.1",
            approval_id="approval.1",
            execution_id="execution.1",
            request_id="request.1",
        )

        changed_approval = build_execution_context_pack(
            state=state,
            candidate=candidate,
            resolved_skill=resolved,
            decision_id="decision.1",
            trace_ref="trace.1",
            approval_id="approval.2",
            execution_id="execution.1",
            request_id="request.1",
        )
        changed_execution = build_execution_context_pack(
            state=state,
            candidate=candidate,
            resolved_skill=resolved,
            decision_id="decision.1",
            trace_ref="trace.1",
            approval_id="approval.1",
            execution_id="execution.2",
            request_id="request.1",
        )
        changed_request = build_execution_context_pack(
            state=state,
            candidate=candidate,
            resolved_skill=resolved,
            decision_id="decision.1",
            trace_ref="trace.1",
            approval_id="approval.1",
            execution_id="execution.1",
            request_id="request.2",
        )

        self.assertNotEqual(base.context_pack_id, changed_approval.context_pack_id)
        self.assertNotEqual(base.context_pack_id, changed_execution.context_pack_id)
        self.assertNotEqual(base.context_pack_id, changed_request.context_pack_id)

    def test_context_pack_does_not_dump_unrelated_state(self) -> None:
        state = _state()
        state.work_items.append(
            WorkItem(work_item_id="work.unrelated", title="Unrelated work item")
        )
        state.risks.append(
            Risk(
                risk_id="risk.unrelated",
                kind="noise",
                description="Unrelated risk.",
                applies_to_refs=["work.unrelated"],
            )
        )

        pack = build_execution_context_pack(
            state=state,
            candidate=_candidate(),
            resolved_skill=_resolved_skill(),
        )

        work_ids = [item["work_item_id"] for item in pack.relevant_state["work_items"]]
        risk_ids = [item["risk_id"] for item in pack.relevant_state["risks"]]
        self.assertNotIn("work.unrelated", work_ids)
        self.assertNotIn("risk.unrelated", risk_ids)
        encoded = json.dumps(pack.to_payload())
        self.assertNotIn("Unrelated work item", encoded)
        self.assertNotIn("Unrelated risk.", encoded)

    def test_max_items_per_section_limits_context_size(self) -> None:
        state = _state()
        state.commitments.append(Commitment(commitment_id="commitment.2", title="Second"))
        state.commitments.append(Commitment(commitment_id="commitment.3", title="Third"))

        pack = build_execution_context_pack(
            state=state,
            candidate=_candidate(),
            resolved_skill=_resolved_skill(),
            max_items_per_section=1,
        )

        self.assertEqual(len(pack.relevant_state["commitments"]), 1)

    def test_mismatched_resolved_skill_action_type_is_rejected(self) -> None:
        resolved = _resolved_skill()
        resolved.action_type = "intent.execute"

        with self.assertRaisesRegex(ValueError, "action_type"):
            build_execution_context_pack(
                state=_state(),
                candidate=_candidate(),
                resolved_skill=resolved,
            )

    def test_mismatched_resolved_skill_candidate_id_is_rejected(self) -> None:
        resolved = _resolved_skill()
        resolved.metadata["candidate_id"] = "candidate.other"

        with self.assertRaisesRegex(ValueError, "candidate_id"):
            build_execution_context_pack(
                state=_state(),
                candidate=_candidate(),
                resolved_skill=resolved,
            )

    def test_builder_does_not_mutate_inputs(self) -> None:
        state = _state()
        candidate = _candidate()
        resolved = _resolved_skill()
        before_state = copy.deepcopy(state.to_payload())
        before_candidate = copy.deepcopy(candidate.to_payload())
        before_resolved = copy.deepcopy(resolved.to_payload())

        build_execution_context_pack(
            state=state,
            candidate=candidate,
            resolved_skill=resolved,
        )

        self.assertEqual(state.to_payload(), before_state)
        self.assertEqual(candidate.to_payload(), before_candidate)
        self.assertEqual(resolved.to_payload(), before_resolved)


def _state() -> GeneralDecisionState:
    return GeneralDecisionState(
        state_id="world.general",
        signals=[
            Signal(
                signal_id="signal.1",
                source="manual",
                kind="work_item",
                summary="Work arrived.",
                subject_ref="work.1",
            )
        ],
        observations=[
            GenericObservation(
                observation_id="obs.1",
                kind=ObservationKind.WORK_ITEM,
                source=ObservationSource(provider="manual"),
                subject=ObservationSubject(
                    subject_id="work.1",
                    subject_type="work_item",
                    title="Review incoming work",
                ),
                summary="A work item needs triage.",
            )
        ],
        intents=[
            Intent(
                intent_id="intent.1",
                summary="Handle incoming work.",
                target_refs=["work.1"],
            )
        ],
        commitments=[
            Commitment(
                commitment_id="commitment.1",
                title="Fixed commitment",
                status="active",
            )
        ],
        work_items=[
            WorkItem(
                work_item_id="work.1",
                title="Review incoming work",
                urgency="medium",
            )
        ],
        capabilities=[
            Capability(
                capability_id="work_item_triage",
                provider="codex",
                scope="triage",
                requires_confirmation=False,
            )
        ],
        constraints=[
            Constraint(
                constraint_id="constraint.time",
                kind="time_window",
                description="Keep action bounded.",
                applies_to_refs=["work.1"],
            )
        ],
        risks=[
            Risk(
                risk_id="risk.stale_work",
                kind="delay",
                description="Work may get stale.",
                applies_to_refs=["work.1"],
            )
        ],
        outcomes=[
            OutcomeRecord(
                outcome_id="outcome.1",
                decision_id="decision.1",
                trace_ref="trace.1",
                candidate_id="candidate.item.triage.work.1",
                execution_ref="execution.1",
                protocol_status="success",
                task_status="success",
            )
        ],
    )


def _candidate() -> GenericCandidate:
    return GenericCandidate(
        candidate_id="candidate.item.triage.work.1",
        action_type="item.triage",
        intent="Triage the incoming work item.",
        target_refs=["work.1"],
        required_capability="work_item_triage",
        estimated_cost=EstimatedCost(time_minutes=5, attention="low"),
        risk_profile=RiskProfile(level="low"),
        expected_state_delta=ExpectedStateDelta(updates_refs=["work.1"]),
        execution_boundary=ExecutionBoundary(
            mode="skill_resolution",
            required_capability="work_item_triage",
            requires_confirmation=False,
            side_effect_class="read_only",
        ),
        constraints_triggered=[
            {
                "constraint_id": "constraint.time",
                "description": "Keep action bounded.",
            }
        ],
        side_effect_class="low",
        requires_confirmation=False,
        why_available=["Open work item can be triaged."],
        availability_status="available",
    )


def _resolved_skill() -> ResolvedSkill:
    return ResolvedSkill(
        executor_id="codex",
        skill_id="work_item.triage.codex",
        action_type="item.triage",
        capability_id="work_item_triage",
        side_effect_class="read_only",
        requires_confirmation=False,
        resolution_reason="matched action_type and capability",
        metadata={
            "candidate_id": "candidate.item.triage.work.1",
            "target_refs": ["work.1"],
        },
        instructions=["Do not edit files.", "Return a concise triage summary."],
        input_schema={"type": "context_pack.v1"},
        output_schema={"type": "triage_report.v1"},
    )


if __name__ == "__main__":
    unittest.main()
