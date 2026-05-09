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
from examples.decision_hub_demo.general_approval import (
    build_general_approval_artifact,
    build_general_approval_bridge,
)
from examples.decision_hub_demo.general_adapter import (
    GeneralDecisionHubResult,
    decision_hub_general_guidance,
    run_general_read_only_path,
)
from examples.decision_hub_demo.run_demo import (
    build_general_approval_artifact as build_run_demo_general_approval_artifact,
    main as run_demo_main,
)


NOW = datetime(2026, 4, 17, 6, 0, tzinfo=timezone.utc)


class DecisionHubGeneralApprovalTests(unittest.TestCase):
    def test_general_selected_candidate_becomes_pending_approval(self) -> None:
        result = run_general_read_only_path(now=NOW)

        bridge = build_general_approval_bridge(result, now=NOW)
        payload = bridge.to_payload()

        self.assertEqual(payload["status"], "approval_required")
        self.assertEqual(payload["decision_id"], result.policy_result.checkpoint.decision_id)
        self.assertEqual(payload["trace_ref"], result.policy_result.checkpoint.trace_ref)
        self.assertEqual(
            payload["selected_candidate_id"],
            result.policy_result.checkpoint.selected_candidate_id,
        )
        self.assertEqual(payload["approval"]["status"], "pending")
        self.assertFalse(payload["approval"]["execution_allowed"])
        self.assertFalse(payload["execution_allowed"])
        self.assertIsNone(payload["execution"])
        self.assertFalse(payload["state_updated"])

    def test_general_confirmation_payload_is_legacy_compatible_and_traceable(self) -> None:
        result = run_general_read_only_path(now=NOW)

        bridge = build_general_approval_bridge(result, now=NOW)
        request = bridge.confirmation_request

        self.assertIsNotNone(request)
        assert request is not None
        self.assertTrue(request["confirmation_id"].startswith("confirm."))
        self.assertEqual(request["decision_id"], result.policy_result.checkpoint.decision_id)
        self.assertEqual(request["trace_ref"], result.policy_result.checkpoint.trace_ref)
        self.assertEqual(request["candidate_id"], result.policy_result.checkpoint.selected_candidate_id)
        self.assertEqual(request["selected_action"], bridge.selected_candidate.action_type)
        self.assertEqual(
            request["options"],
            [
                {"key": "1", "value": "confirm"},
                {"key": "2", "value": "reject"},
                {"key": "3", "value": "details"},
            ],
        )
        self.assertFalse(request["execution_allowed"])
        self.assertIsNone(request["execution"])

    def test_general_approval_artifact_is_json_serializable(self) -> None:
        result = run_general_read_only_path(now=NOW)

        artifact = build_general_approval_artifact(result, now=NOW)

        self.assertEqual(artifact["path_type"], "read_only_general_approval")
        self.assertEqual(artifact["generated_by"], "general_approval_bridge")
        self.assertEqual(artifact["decision_id"], result.policy_result.checkpoint.decision_id)
        self.assertEqual(artifact["trace_ref"], result.policy_result.checkpoint.trace_ref)
        self.assertEqual(
            artifact["selected_candidate_id"],
            result.policy_result.checkpoint.selected_candidate_id,
        )
        self.assertIsNotNone(artifact["approval"])
        self.assertIsNotNone(artifact["confirmation_request"])
        self.assertFalse(artifact["execution_allowed"])
        self.assertIsNone(artifact["execution"])
        self.assertFalse(artifact["state_updated"])
        self.assertEqual(artifact["created_at"], "2026-04-17T06:00:00Z")
        self.assertEqual(
            artifact["approval_bridge"]["selected_candidate_id"],
            result.policy_result.checkpoint.selected_candidate_id,
        )
        json.dumps(artifact)

    def test_fallback_approval_id_is_stable_and_decision_scoped(self) -> None:
        first = _result_for_candidate(
            _candidate(
                "candidate.capability.use.shared",
                "capability.use",
                requires_confirmation=True,
            ),
            decision_id="decision.general.one",
        )
        first.policy_result.checkpoint.approval = None
        second = _result_for_candidate(
            _candidate(
                "candidate.capability.use.shared",
                "capability.use",
                requires_confirmation=True,
            ),
            decision_id="decision.general.one",
        )
        second.policy_result.checkpoint.approval = None
        third = _result_for_candidate(
            _candidate(
                "candidate.capability.use.shared",
                "capability.use",
                requires_confirmation=True,
            ),
            decision_id="decision.general.two",
        )
        third.policy_result.checkpoint.approval = None

        first_id = build_general_approval_bridge(first, now=NOW).approval.approval_id
        second_id = build_general_approval_bridge(second, now=NOW).approval.approval_id
        third_id = build_general_approval_bridge(third, now=NOW).approval.approval_id

        self.assertEqual(first_id, second_id)
        self.assertNotEqual(first_id, third_id)
        self.assertIn("decision_general_one", first_id)
        self.assertIn("candidate_capability_use_shared", first_id)

    def test_non_confirmation_candidate_does_not_create_pending_approval(self) -> None:
        result = _result_for_candidate(
            _candidate(
                "candidate.item.triage.no_confirmation",
                "item.triage",
                requires_confirmation=False,
            )
        )

        bridge = build_general_approval_bridge(result, now=NOW)
        payload = bridge.to_payload()
        artifact = build_general_approval_artifact(result, now=NOW)

        self.assertEqual(payload["status"], "approval_not_required")
        self.assertIsNone(payload["approval"])
        self.assertIsNone(payload["confirmation_request"])
        self.assertFalse(payload["execution_allowed"])
        self.assertIsNone(payload["execution"])
        self.assertFalse(payload["state_updated"])
        self.assertIsNone(artifact["approval"])
        self.assertIsNone(artifact["confirmation_request"])
        self.assertFalse(artifact["execution_allowed"])
        self.assertIsNone(artifact["execution"])
        self.assertFalse(artifact["state_updated"])
        self.assertNotIn("execute.request", json.dumps(artifact).lower())

    def test_run_demo_general_approval_is_read_only(self) -> None:
        stdout = io.StringIO()
        with patch(
            "examples.decision_hub_demo.run_demo.DecisionControlLoop.handle_recommendation",
            side_effect=AssertionError("General approval must not enter legacy confirmation loop"),
        ), patch(
            "examples.decision_hub_demo.sdep_executor.create_default_sdep_executor",
            side_effect=AssertionError("General approval must not create an SDEP executor"),
        ), patch(
            "examples.decision_hub_demo.sdep_executor.execution_request_to_intent",
            side_effect=AssertionError("General approval must not create ExecutionIntent"),
        ), redirect_stdout(stdout):
            run_demo_main(["--general-approval"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["path_type"], "read_only_general_approval")
        self.assertEqual(payload["generated_by"], "general_approval_bridge")
        self.assertFalse(payload["execution_allowed"])
        self.assertIsNone(payload["execution"])
        self.assertFalse(payload["state_updated"])
        self.assertIsNone(payload["approval_bridge"]["execution"])
        self.assertFalse(payload["approval_bridge"]["state_updated"])

    def test_run_demo_general_approval_builder_uses_general_path(self) -> None:
        artifact = build_run_demo_general_approval_artifact(now=NOW)
        encoded = json.dumps(artifact).lower()

        self.assertEqual(artifact["path_type"], "read_only_general_approval")
        self.assertIn("decision.general.decision_hub", encoded)
        self.assertNotIn("execute.request", encoded)
        self.assertNotIn('"execution_path": "sdep', encoded)


def _result_for_candidate(
    candidate: GenericCandidate,
    *,
    decision_id: str = "decision.general.approval_test",
) -> GeneralDecisionHubResult:
    state = GeneralDecisionState(state_id="world.general.approval_test")
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
        target_refs=["target.general.approval_test"],
        estimated_cost=EstimatedCost(time_minutes=5, attention="low"),
        risk_profile=RiskProfile(level="low", uncertainty="low"),
        reversibility="high",
        requires_confirmation=requires_confirmation,
        expected_state_delta=ExpectedStateDelta(
            updates_refs=["target.general.approval_test"],
            summary="Expected state update.",
        ),
        execution_boundary=ExecutionBoundary(
            mode="capability" if requires_confirmation else "none",
            requires_confirmation=requires_confirmation,
            side_effect_class="external" if requires_confirmation else "none",
        ),
        why_available=["Candidate is available for approval bridge test."],
        side_effect_class="external" if requires_confirmation else "none",
        availability_status="available",
    )


if __name__ == "__main__":
    unittest.main()
