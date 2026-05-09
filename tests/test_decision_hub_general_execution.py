from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from unittest.mock import patch

from spice.decision.general import (
    EstimatedCost,
    ExecutionBoundary,
    ExpectedStateDelta,
    GeneralDecisionState,
    GenericCandidate,
    GenericPolicyAdapter,
    RiskProfile,
)
from spice.protocols.sdep import SDEPExecuteRequest
from examples.decision_hub_demo.general_adapter import (
    GeneralDecisionHubResult,
    decision_hub_general_guidance,
    run_general_read_only_path,
)
from examples.decision_hub_demo.general_approval import build_general_approval_bridge
from examples.decision_hub_demo.general_execution import (
    approve_general_approval,
    build_general_execution_artifact,
    build_general_execution_plan,
)
from examples.decision_hub_demo.run_demo import (
    build_general_execution_artifact as build_run_demo_general_execution_artifact,
    main as run_demo_main,
)


NOW = datetime(2026, 4, 17, 6, 0, tzinfo=timezone.utc)


class DecisionHubGeneralExecutionTests(unittest.TestCase):
    def test_pending_approval_does_not_create_execution_plan(self) -> None:
        result = run_general_read_only_path(now=NOW)

        plan = build_general_execution_plan(result, now=NOW)
        payload = plan.to_payload()

        self.assertEqual(payload["status"], "approval_required")
        self.assertFalse(payload["execution_allowed"])
        self.assertFalse(payload["executed"])
        self.assertIsNone(payload["execution"])
        self.assertFalse(payload["state_updated"])
        self.assertIsNone(payload["execution_intent"])
        self.assertIsNone(payload["sdep_request"])

    def test_approved_general_candidate_creates_planned_sdep_request_only(self) -> None:
        result = run_general_read_only_path(now=NOW)
        bridge = build_general_approval_bridge(result, now=NOW)
        assert bridge.approval is not None
        approved = approve_general_approval(bridge.approval, now=NOW)

        artifact = build_general_execution_artifact(result, approval=approved, now=NOW)

        self.assertEqual(artifact["path_type"], "read_only_general_execution_plan")
        self.assertEqual(artifact["generated_by"], "general_execution_planner")
        self.assertEqual(artifact["status"], "planned")
        self.assertEqual(artifact["candidate_id"], bridge.selected_candidate_id)
        self.assertEqual(artifact["execution_status"], "planned_not_executed")
        self.assertTrue(artifact["approved"])
        self.assertTrue(artifact["execution_allowed"])
        self.assertFalse(artifact["executed"])
        self.assertIsNone(artifact["execution"])
        self.assertIsNone(artifact["outcome"])
        self.assertFalse(artifact["state_updated"])
        self.assertEqual(artifact["decision_id"], bridge.decision_id)
        self.assertEqual(artifact["trace_ref"], bridge.trace_ref)
        self.assertEqual(artifact["selected_candidate_id"], bridge.selected_candidate_id)
        self.assertEqual(artifact["approval_id"], approved.approval_id)
        self.assertEqual(artifact["skill_resolution_status"], "resolved")
        self.assertTrue(artifact["skill_id"])
        self.assertTrue(artifact["executor_id"])
        self.assertTrue(artifact["context_pack_id"])
        self.assertIsInstance(artifact["resolved_skill"], dict)
        self.assertIsInstance(artifact["context_pack"], dict)
        self.assertEqual(artifact["resolved_skill"]["skill_id"], artifact["skill_id"])
        self.assertIn("instructions", artifact["resolved_skill"])
        self.assertIn("input_schema", artifact["resolved_skill"])
        self.assertIn("output_schema", artifact["resolved_skill"])
        self.assertTrue(artifact["resolved_skill"]["instructions"])
        self.assertEqual(artifact["context_pack"]["context_pack_id"], artifact["context_pack_id"])
        self.assertEqual(artifact["context_pack"]["candidate_id"], bridge.selected_candidate_id)
        self.assertEqual(artifact["context_pack"]["decision_id"], bridge.decision_id)
        self.assertTrue(artifact["context_pack"]["task"])
        self.assertTrue(artifact["context_pack"]["why_now"])
        self.assertTrue(artifact["context_pack"]["do_not"])
        self.assertTrue(artifact["context_pack"]["expected_output"])
        self.assertEqual(
            artifact["context_pack"]["return_schema"],
            artifact["resolved_skill"]["output_schema"],
        )

        intent = artifact["execution_intent"]
        sdep_request = artifact["sdep_request"]
        self.assertIsNotNone(intent)
        self.assertIsNotNone(sdep_request)
        assert isinstance(intent, dict)
        assert isinstance(sdep_request, dict)
        self.assertEqual(artifact["execution_id"], intent["id"])
        self.assertEqual(intent["status"], "planned")
        self.assertEqual(intent["input_payload"]["approval_id"], approved.approval_id)
        self.assertEqual(intent["input_payload"]["candidate_id"], bridge.selected_candidate_id)
        self.assertEqual(intent["input_payload"]["skill_id"], artifact["skill_id"])
        self.assertEqual(intent["input_payload"]["executor_id"], artifact["executor_id"])
        self.assertEqual(intent["input_payload"]["context_pack_id"], artifact["context_pack_id"])
        self.assertEqual(
            intent["input_payload"]["context_pack"]["context_pack_id"],
            artifact["context_pack_id"],
        )
        self.assertEqual(intent["parameters"]["resolved_skill"]["skill_id"], artifact["skill_id"])
        self.assertEqual(
            intent["input_payload"]["context_pack"]["return_schema"],
            artifact["resolved_skill"]["output_schema"],
        )
        self.assertTrue(intent["operation"]["name"].startswith("spice.general."))
        self.assertEqual(intent["operation"]["mode"], "sync")
        self.assertEqual(sdep_request["message_type"], "execute.request")
        self.assertEqual(sdep_request["idempotency_key"], intent["id"])
        self.assertEqual(sdep_request["execution"]["action_type"], intent["operation"]["name"])
        self.assertEqual(sdep_request["execution"]["mode"], "sync")
        self.assertEqual(sdep_request["traceability"]["spice_decision_id"], bridge.decision_id)
        self.assertEqual(sdep_request["traceability"]["trace_ref"], bridge.trace_ref)
        self.assertEqual(sdep_request["traceability"]["candidate_id"], bridge.selected_candidate_id)
        self.assertEqual(sdep_request["traceability"]["approval_id"], approved.approval_id)
        self.assertTrue(sdep_request["metadata"]["planning_only"])
        self.assertEqual(sdep_request["execution"]["input"]["skill_id"], artifact["skill_id"])
        self.assertEqual(
            sdep_request["execution"]["input"]["context_pack"]["context_pack_id"],
            artifact["context_pack_id"],
        )
        self.assertEqual(sdep_request["execution"]["metadata"]["skill_id"], artifact["skill_id"])
        self.assertEqual(
            sdep_request["execution"]["metadata"]["context_pack_id"],
            artifact["context_pack_id"],
        )
        json.dumps(artifact)

    def test_sdep_request_round_trips_through_protocol_model(self) -> None:
        result = run_general_read_only_path(now=NOW)
        bridge = build_general_approval_bridge(result, now=NOW)
        assert bridge.approval is not None
        approved = approve_general_approval(bridge.approval, now=NOW)

        artifact = build_general_execution_artifact(result, approval=approved, now=NOW)
        request = SDEPExecuteRequest.from_dict(artifact["sdep_request"])
        wire_payload = request.to_dict()

        self.assertEqual(wire_payload["message_type"], "execute.request")
        self.assertEqual(wire_payload["idempotency_key"], artifact["execution_id"])
        self.assertEqual(wire_payload["traceability"]["execution_id"], artifact["execution_id"])
        self.assertEqual(wire_payload["traceability"]["spice_decision_id"], bridge.decision_id)
        self.assertEqual(wire_payload["traceability"]["trace_ref"], bridge.trace_ref)
        self.assertEqual(wire_payload["traceability"]["candidate_id"], bridge.selected_candidate_id)
        self.assertEqual(wire_payload["traceability"]["approval_id"], approved.approval_id)
        self.assertTrue(wire_payload["metadata"]["planning_only"])
        self.assertTrue(wire_payload["execution"]["metadata"]["planning_only"])
        self.assertTrue(wire_payload["execution"]["input"]["skill_id"])
        self.assertEqual(
            wire_payload["execution"]["input"]["context_pack"]["context_pack_id"],
            artifact["context_pack_id"],
        )

    def test_explicit_approval_must_match_decision_trace_candidate_and_approval_id(self) -> None:
        result = run_general_read_only_path(now=NOW)
        bridge = build_general_approval_bridge(result, now=NOW)
        assert bridge.approval is not None
        approved = approve_general_approval(bridge.approval, now=NOW)

        cases = [
            ("decision_id", "decision.other", "decision_id mismatch"),
            ("candidate_id", "candidate.other", "candidate_id mismatch"),
            ("approval_id", "approval.other", "approval_id mismatch"),
            ("metadata", {"trace_ref": "trace.other"}, "trace_ref mismatch"),
        ]
        for field_name, value, error_token in cases:
            with self.subTest(field=field_name):
                payload = approved.to_payload()
                payload[field_name] = value
                with self.assertRaisesRegex(ValueError, error_token):
                    build_general_execution_plan(result, approval=payload, now=NOW)

    def test_explicit_unapproved_approval_is_rejected(self) -> None:
        result = run_general_read_only_path(now=NOW)
        bridge = build_general_approval_bridge(result, now=NOW)
        assert bridge.approval is not None

        with self.assertRaisesRegex(ValueError, "approval must be approved"):
            build_general_execution_plan(result, approval=bridge.approval, now=NOW)

    def test_execution_id_is_stable_and_bound_to_attribution(self) -> None:
        result = run_general_read_only_path(now=NOW)
        bridge = build_general_approval_bridge(result, now=NOW)
        assert bridge.approval is not None
        approved = approve_general_approval(bridge.approval, now=NOW)

        first = build_general_execution_artifact(result, approval=approved, now=NOW)
        second = build_general_execution_artifact(result, approval=approved, now=NOW)
        self.assertEqual(first["execution_id"], second["execution_id"])

        other_candidate_result = _result_for_candidate(
            _candidate(
                "candidate.item.triage.other",
                "item.triage",
                requires_confirmation=True,
            ),
            decision_id="decision.general.execution_test.other_candidate",
        )
        other_bridge = build_general_approval_bridge(other_candidate_result, now=NOW)
        assert other_bridge.approval is not None
        other_approved = approve_general_approval(other_bridge.approval, now=NOW)
        other_artifact = build_general_execution_artifact(
            other_candidate_result,
            approval=other_approved,
            now=NOW,
        )
        self.assertNotEqual(first["approval_id"], other_artifact["approval_id"])
        self.assertNotEqual(first["candidate_id"], other_artifact["candidate_id"])
        self.assertNotEqual(first["execution_id"], other_artifact["execution_id"])

    def test_non_confirmation_candidate_can_be_planned_without_approval(self) -> None:
        result = _result_for_candidate(
            _candidate(
                "candidate.item.triage.no_confirmation",
                "item.triage",
                requires_confirmation=False,
            )
        )

        artifact = build_general_execution_artifact(result, now=NOW)

        self.assertEqual(artifact["status"], "planned")
        self.assertEqual(artifact["execution_status"], "planned_not_executed")
        self.assertTrue(artifact["approved"])
        self.assertIsNone(artifact["approval_id"])
        self.assertTrue(artifact["execution_allowed"])
        self.assertIsNotNone(artifact["execution_id"])
        self.assertIsNotNone(artifact["execution_intent"])
        self.assertIsNotNone(artifact["sdep_request"])
        self.assertFalse(artifact["executed"])
        self.assertIsNone(artifact["outcome"])
        self.assertFalse(artifact["state_updated"])

    def test_approval_required_artifact_has_no_execution_payload(self) -> None:
        result = run_general_read_only_path(now=NOW)

        artifact = build_general_execution_artifact(result, now=NOW)

        self.assertEqual(artifact["status"], "approval_required")
        self.assertEqual(artifact["execution_status"], "approval_required")
        self.assertFalse(artifact["approved"])
        self.assertFalse(artifact["execution_allowed"])
        self.assertIsNone(artifact["execution_id"])
        self.assertEqual(artifact["skill_resolution_status"], "unresolved")
        self.assertIsNone(artifact["skill_id"])
        self.assertIsNone(artifact["executor_id"])
        self.assertIsNone(artifact["context_pack_id"])
        self.assertIsNone(artifact["resolved_skill"])
        self.assertIsNone(artifact["context_pack"])
        self.assertIsNone(artifact["execution_intent"])
        self.assertIsNone(artifact["sdep_request"])
        self.assertIsNone(artifact["execution"])
        self.assertIsNone(artifact["outcome"])
        self.assertFalse(artifact["executed"])
        self.assertFalse(artifact["state_updated"])
        json.dumps(artifact)

    def test_run_demo_general_execution_plan_is_read_only(self) -> None:
        stdout = io.StringIO()
        with patch(
            "examples.decision_hub_demo.run_demo.DecisionControlLoop.handle_recommendation",
            side_effect=AssertionError("General execution plan must not enter legacy confirmation loop"),
        ), patch(
            "examples.decision_hub_demo.sdep_executor.create_default_sdep_executor",
            side_effect=AssertionError("General execution plan must not create an SDEP executor"),
        ), patch(
            "spice.executors.sdep.SDEPExecutor.execute",
            side_effect=AssertionError("General execution plan must not execute through SDEP"),
        ), redirect_stdout(stdout):
            run_demo_main(["--general-execution-plan"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["path_type"], "read_only_general_execution_plan")
        self.assertEqual(payload["status"], "planned")
        self.assertEqual(payload["execution_status"], "planned_not_executed")
        self.assertIsNotNone(payload["sdep_request"])
        self.assertFalse(payload["executed"])
        self.assertIsNone(payload["execution"])
        self.assertIsNone(payload["outcome"])
        self.assertFalse(payload["state_updated"])

    def test_run_demo_general_execution_builder_uses_general_path(self) -> None:
        artifact = build_run_demo_general_execution_artifact(now=NOW)
        encoded = json.dumps(artifact).lower()

        self.assertEqual(artifact["path_type"], "read_only_general_execution_plan")
        self.assertIn("decision.general.decision_hub", encoded)
        self.assertIn("execute.request", encoded)
        self.assertNotIn("decision_hub.delegate_to_executor", encoded)
        self.assertNotIn("execution_result", encoded)
        self.assertIn("outcome", artifact)
        self.assertIsNone(artifact["outcome"])


def _result_for_candidate(
    candidate: GenericCandidate,
    *,
    decision_id: str = "decision.general.execution_test",
) -> GeneralDecisionHubResult:
    state = GeneralDecisionState(state_id="world.general.execution_test")
    policy_result = GenericPolicyAdapter(decision_hub_general_guidance()).evaluate(
        state,
        candidates=[candidate],
        decision_id=decision_id,
        trace_ref=f"trace.{decision_id}",
    )
    return GeneralDecisionHubResult(
        observations=[],
        state=state,
        candidates=[candidate],
        policy_result=policy_result,
    )


def _candidate(
    candidate_id: str,
    action_type: str,
    *,
    requires_confirmation: bool,
) -> GenericCandidate:
    return GenericCandidate(
        candidate_id=candidate_id,
        action_type=action_type,
        intent=f"Test {action_type}.",
        target_refs=["target.general.execution_test"],
        estimated_cost=EstimatedCost(time_minutes=5, attention="low"),
        risk_profile=RiskProfile(level="low", uncertainty="low"),
        reversibility="high",
        requires_confirmation=requires_confirmation,
        expected_state_delta=ExpectedStateDelta(
            updates_refs=["target.general.execution_test"],
            summary="Expected state update.",
        ),
        execution_boundary=ExecutionBoundary(
            mode="capability" if requires_confirmation else "sync",
            requires_confirmation=requires_confirmation,
            side_effect_class="state_change",
        ),
        why_available=["Candidate is available for execution planning test."],
        side_effect_class="state_change",
        availability_status="available",
    )


if __name__ == "__main__":
    unittest.main()
