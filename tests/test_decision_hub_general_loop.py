from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from unittest.mock import patch

from examples.decision_hub_demo.general_loop import (
    build_general_loop_artifact,
    render_general_loop_text,
)
from examples.decision_hub_demo.run_demo import (
    build_general_loop_artifact as build_run_demo_general_loop_artifact,
    main as run_demo_main,
)


NOW = datetime(2026, 4, 17, 6, 0, tzinfo=timezone.utc)


class DecisionHubGeneralLoopTests(unittest.TestCase):
    def test_full_loop_artifact_links_all_read_only_steps(self) -> None:
        artifact = build_general_loop_artifact(now=NOW, use_bars=False)

        self.assertEqual(artifact["path_type"], "read_only_general_full_loop")
        self.assertEqual(artifact["generated_by"], "decision_hub_demo.general_loop")
        self.assertEqual(artifact["status"], "full_loop_rendered")
        self.assertEqual(artifact["loop_status"], "completed_read_only")
        self.assertTrue(artifact["read_only"])
        self.assertFalse(artifact["executor_called"])
        self.assertFalse(artifact["executed"])
        self.assertIsNone(artifact["execution"])
        self.assertFalse(artifact["sdep_request_sent"])
        self.assertFalse(artifact["persisted"])
        self.assertTrue(artifact["state_snapshot_updated"])
        self.assertEqual(artifact["update_mode"], "read_only_snapshot")
        self.assertTrue(artifact["request_id"])
        self.assertIn("state_before_summary", artifact)
        self.assertIn("state_after_summary", artifact)
        self.assertIn("state_before", artifact)
        self.assertIn("state_after", artifact)
        self.assertIn("observations", artifact)
        self.assertIn("candidates", artifact)
        self.assertIn("candidate_summary", artifact)
        self.assertIn("compare_payload", artifact)
        self.assertIn("approval_artifact", artifact)
        self.assertIn("execution_artifact", artifact)
        self.assertIn("outcome_artifact", artifact)
        self.assertIn("state_feedback_artifact", artifact)
        self.assertTrue(artifact["candidate_summary"]["selected_present"])
        self.assertEqual(
            artifact["flow"],
            [
                "observations",
                "general_state",
                "generic_candidates",
                "policy_decision",
                "approval_checkpoint",
                "skill_resolution",
                "context_pack",
                "sdep_request_plan",
                "sdep_response_fixture",
                "outcome_observation",
                "state_feedback_snapshot",
            ],
        )

        self.assertEqual(artifact["decision"]["compare_payload"]["decision_id"], artifact["decision_id"])
        self.assertEqual(artifact["approval"]["decision_id"], artifact["decision_id"])
        self.assertEqual(artifact["execution_plan"]["decision_id"], artifact["decision_id"])
        self.assertEqual(artifact["execution_plan"]["skill_id"], artifact["skill_id"])
        self.assertEqual(artifact["execution_plan"]["executor_id"], artifact["executor_id"])
        self.assertEqual(artifact["execution_plan"]["context_pack_id"], artifact["context_pack_id"])
        self.assertEqual(artifact["resolved_skill"]["skill_id"], artifact["skill_id"])
        self.assertEqual(artifact["resolved_skill"]["executor_id"], artifact["executor_id"])
        self.assertEqual(artifact["context_pack"]["context_pack_id"], artifact["context_pack_id"])
        self.assertEqual(artifact["context_pack"]["decision_id"], artifact["decision_id"])
        self.assertEqual(artifact["context_pack"]["trace_ref"], artifact["trace_ref"])
        self.assertEqual(artifact["context_pack"]["candidate_id"], artifact["selected_candidate_id"])
        self.assertEqual(artifact["outcome_return"]["decision_id"], artifact["decision_id"])
        self.assertEqual(artifact["state_feedback"]["decision_id"], artifact["decision_id"])
        self.assertEqual(artifact["execution_plan"]["execution_id"], artifact["execution_id"])
        self.assertEqual(artifact["outcome_return"]["execution_id"], artifact["execution_id"])
        self.assertEqual(artifact["state_feedback"]["execution_id"], artifact["execution_id"])
        self.assertEqual(artifact["outcome_return"]["outcome_id"], artifact["outcome_id"])
        self.assertEqual(artifact["state_feedback"]["outcome_id"], artifact["outcome_id"])
        json.dumps(artifact)

    def test_full_loop_skill_and_context_ids_are_consistent_across_handoff(self) -> None:
        artifact = build_general_loop_artifact(now=NOW, use_bars=False)
        sdep_request = artifact["execution_plan"]["sdep_request"]

        self.assertEqual(artifact["skill_id"], artifact["resolved_skill"]["skill_id"])
        self.assertEqual(artifact["executor_id"], artifact["resolved_skill"]["executor_id"])
        self.assertEqual(artifact["context_pack_id"], artifact["context_pack"]["context_pack_id"])
        self.assertEqual(artifact["execution_artifact"]["skill_id"], artifact["skill_id"])
        self.assertEqual(artifact["execution_artifact"]["context_pack_id"], artifact["context_pack_id"])

        self.assertEqual(sdep_request["execution"]["input"]["skill_hint"]["skill_id"], artifact["skill_id"])
        self.assertEqual(
            sdep_request["execution"]["input"]["context_pack"]["context_pack_id"],
            artifact["context_pack_id"],
        )
        self.assertEqual(artifact["context_pack"]["decision_id"], artifact["decision_id"])
        self.assertEqual(artifact["context_pack"]["trace_ref"], artifact["trace_ref"])
        self.assertEqual(artifact["context_pack"]["candidate_id"], artifact["selected_candidate_id"])
        self.assertEqual(artifact["context_pack"]["approval_id"], artifact["approval_id"])
        self.assertEqual(artifact["context_pack"]["execution_id"], artifact["execution_id"])
        self.assertEqual(artifact["context_pack"]["request_id"], artifact["request_id"])

    def test_full_loop_ids_are_consistent_across_segments(self) -> None:
        artifact = build_general_loop_artifact(now=NOW, use_bars=False)
        decision_id = artifact["decision_id"]
        trace_ref = artifact["trace_ref"]
        candidate_id = artifact["selected_candidate_id"]
        approval_id = artifact["approval_id"]
        execution_id = artifact["execution_id"]
        request_id = artifact["request_id"]
        outcome_id = artifact["outcome_id"]

        self.assertEqual(artifact["decision"]["compare_payload"]["decision_id"], decision_id)
        self.assertEqual(artifact["approval"]["decision_id"], decision_id)
        self.assertEqual(artifact["execution_plan"]["decision_id"], decision_id)
        self.assertEqual(artifact["outcome_return"]["decision_id"], decision_id)
        self.assertEqual(artifact["state_feedback"]["decision_id"], decision_id)

        self.assertEqual(artifact["decision"]["compare_payload"]["trace_ref"], trace_ref)
        self.assertEqual(artifact["approval"]["trace_ref"], trace_ref)
        self.assertEqual(artifact["execution_plan"]["trace_ref"], trace_ref)
        self.assertEqual(artifact["outcome_return"]["trace_ref"], trace_ref)
        self.assertEqual(artifact["state_feedback"]["trace_ref"], trace_ref)

        self.assertEqual(artifact["approval"]["selected_candidate_id"], candidate_id)
        self.assertEqual(artifact["approval"]["approval"]["candidate_id"], candidate_id)
        self.assertEqual(artifact["execution_plan"]["candidate_id"], candidate_id)
        self.assertEqual(artifact["context_pack"]["candidate_id"], candidate_id)
        self.assertEqual(artifact["outcome_return"]["candidate_id"], candidate_id)
        self.assertEqual(artifact["state_feedback"]["candidate_id"], candidate_id)

        self.assertEqual(artifact["approval"]["approval"]["approval_id"], approval_id)
        self.assertEqual(artifact["execution_plan"]["approval_id"], approval_id)
        self.assertEqual(artifact["outcome_return"]["approval_id"], approval_id)

        self.assertEqual(artifact["execution_plan"]["execution_id"], execution_id)
        self.assertEqual(artifact["execution_plan"]["sdep_request"]["traceability"]["execution_id"], execution_id)
        self.assertEqual(artifact["context_pack"]["execution_id"], execution_id)
        self.assertEqual(artifact["outcome_return"]["execution_id"], execution_id)
        self.assertEqual(artifact["state_feedback"]["execution_id"], execution_id)

        self.assertEqual(artifact["execution_plan"]["sdep_request"]["request_id"], request_id)
        self.assertEqual(artifact["context_pack"]["request_id"], request_id)
        self.assertEqual(artifact["outcome_return"]["sdep_response"]["request_id"], request_id)
        self.assertEqual(artifact["outcome_return"]["request_id"], request_id)

        self.assertEqual(artifact["outcome_return"]["outcome_id"], outcome_id)
        self.assertEqual(artifact["state_feedback"]["outcome_id"], outcome_id)
        matching = [
            outcome for outcome in artifact["state_after"]["outcomes"]
            if outcome["outcome_id"] == outcome_id
        ]
        self.assertEqual(len(matching), 1)
        metadata = matching[0]["metadata"]
        self.assertEqual(metadata["request_id"], request_id)
        self.assertEqual(metadata["approval_id"], approval_id)
        self.assertEqual(metadata["protocol_status"], artifact["protocol_status"])
        self.assertEqual(metadata["task_status"], artifact["task_status"])

    def test_full_loop_text_is_screenshot_friendly(self) -> None:
        artifact = build_general_loop_artifact(now=NOW, use_bars=False)
        rendered = render_general_loop_text(artifact)

        self.assertEqual(rendered, artifact["rendered_text"])
        self.assertIn("SPICE DECISION LOOP", rendered)
        self.assertIn("read-only preview: decision -> approval -> skill handoff -> outcome snapshot", rendered)
        self.assertIn("no executor called | no SDEP sent | no state persisted", rendered)
        self.assertIn("0. INPUT SIGNALS", rendered)
        self.assertIn("1. GENERAL STATE", rendered)
        self.assertIn("2. CANDIDATE DECISIONS", rendered)
        self.assertIn("3. SELECTED DECISION", rendered)
        self.assertIn("5. APPROVAL CHECKPOINT", rendered)
        self.assertIn("6. EXECUTION HANDOFF", rendered)
        self.assertIn("7. EXECUTION BOUNDARY", rendered)
        self.assertIn("8. OUTCOME RETURN", rendered)
        self.assertIn("9. STATE FEEDBACK", rendered)
        self.assertIn("10. TRACE", rendered)
        self.assertIn("skill:", rendered)
        self.assertIn("skill_source:", rendered)
        self.assertIn("planned_executor:", rendered)
        self.assertIn("context_pack_id:", rendered)
        self.assertIn("context: compact pack", rendered)
        self.assertIn("task:", rendered)
        self.assertIn("why_now:", rendered)
        self.assertIn("expected_output:", rendered)
        self.assertIn("status: approval_required", rendered)
        self.assertIn("local approval fixture used for preview: true", rendered)
        self.assertNotIn("status: None", rendered)
        self.assertNotIn("executor_id:", rendered)
        self.assertNotIn("executor_hint:", rendered)
        self.assertIn("read_only: true", rendered)
        self.assertIn("sdep_request_sent: false", rendered)
        self.assertIn("executed: false", rendered)
        self.assertIn("persisted: false", rendered)
        self.assertIn("executor_called: false", rendered)
        self.assertNotIn("executed successfully", rendered)

    def test_run_demo_general_full_loop_prints_text(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            run_demo_main(["--general-full-loop", "--no-bars"])

        rendered = stdout.getvalue()

        self.assertIn("SPICE DECISION LOOP", rendered)
        self.assertIn("STATE FEEDBACK", rendered)
        self.assertNotIn('"path_type"', rendered)

    def test_run_demo_general_full_loop_json_prints_artifact(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            run_demo_main(["--general-full-loop-json", "--no-bars"])

        payload = json.loads(stdout.getvalue())

        self.assertEqual(payload["path_type"], "read_only_general_full_loop")
        self.assertIn("rendered_text", payload)
        self.assertEqual(payload["state_feedback"]["update_mode"], "read_only_snapshot")
        self.assertFalse(payload["executor_called"])
        self.assertFalse(payload["persisted"])

    def test_run_demo_general_full_loop_is_read_only(self) -> None:
        stdout = io.StringIO()
        with patch(
            "examples.decision_hub_demo.run_demo.DecisionControlLoop.handle_recommendation",
            side_effect=AssertionError("General full loop must not enter legacy confirmation loop"),
        ), patch(
            "examples.decision_hub_demo.sdep_executor.create_default_sdep_executor",
            side_effect=AssertionError("General full loop must not create an SDEP executor"),
        ), patch(
            "spice.executors.sdep.SDEPExecutor.execute",
            side_effect=AssertionError("General full loop must not execute through SDEP"),
        ), redirect_stdout(stdout):
            run_demo_main(["--general-full-loop"])

        self.assertIn("SPICE DECISION LOOP", stdout.getvalue())

    def test_run_demo_builder_matches_general_loop_builder_contract(self) -> None:
        artifact = build_run_demo_general_loop_artifact(now=NOW, use_bars=False)

        self.assertEqual(artifact["path_type"], "read_only_general_full_loop")
        self.assertEqual(artifact["decision_id"], artifact["state_feedback"]["decision_id"])
        self.assertEqual(artifact["outcome_id"], artifact["state_feedback"]["outcome_id"])


if __name__ == "__main__":
    unittest.main()
