from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import patch

from spice.decision.general import GenericObservation

from examples.decision_hub_demo.general_adapter import run_general_read_only_path
from examples.decision_hub_demo.general_outcome import (
    build_general_outcome_artifact,
    build_general_sdep_response_fixture,
)
from examples.decision_hub_demo.general_state_feedback import (
    build_general_state_feedback,
    build_general_state_feedback_artifact,
)
from examples.decision_hub_demo.run_demo import (
    build_general_execution_artifact,
    build_general_state_feedback_artifact as build_run_demo_general_state_feedback_artifact,
    main as run_demo_main,
)


NOW = datetime(2026, 4, 17, 6, 0, tzinfo=timezone.utc)


class DecisionHubGeneralStateFeedbackTests(unittest.TestCase):
    def test_outcome_observation_applies_to_new_general_state_snapshot(self) -> None:
        result = run_general_read_only_path(now=NOW)
        outcome_artifact = _outcome_artifact()

        artifact = build_general_state_feedback_artifact(
            result.state,
            outcome_artifact,
            now=NOW,
        )

        self.assertEqual(artifact["path_type"], "read_only_general_state_feedback")
        self.assertEqual(artifact["generated_by"], "decision_hub_demo.general_state_feedback")
        self.assertEqual(artifact["decision_id"], outcome_artifact["decision_id"])
        self.assertEqual(artifact["trace_ref"], outcome_artifact["trace_ref"])
        self.assertEqual(artifact["candidate_id"], outcome_artifact["candidate_id"])
        self.assertEqual(artifact["approval_id"], outcome_artifact["approval_id"])
        self.assertEqual(artifact["execution_id"], outcome_artifact["execution_id"])
        self.assertEqual(artifact["outcome_id"], outcome_artifact["outcome_id"])
        self.assertEqual(artifact["protocol_status"], "success")
        self.assertEqual(artifact["task_status"], "success")
        self.assertTrue(artifact["state_updated"])
        self.assertTrue(artifact["state_snapshot_updated"])
        self.assertEqual(artifact["update_mode"], "read_only_snapshot")
        self.assertFalse(artifact["persisted"])
        self.assertFalse(artifact["executor_called"])
        self.assertFalse(artifact["executed"])
        self.assertIsNone(artifact["execution"])
        self.assertEqual(artifact["state_before_summary"]["outcome_count"], 0)
        self.assertEqual(artifact["state_after_summary"]["outcome_count"], 1)
        self.assertEqual(artifact["state_before_summary"]["observation_count"] + 1, artifact["state_after_summary"]["observation_count"])
        self.assertEqual(artifact["state_after"]["outcomes"][0]["outcome_id"], outcome_artifact["outcome_id"])
        self.assertEqual(
            artifact["state_after"]["outcomes"][0]["metadata"]["approval_id"],
            outcome_artifact["approval_id"],
        )
        self.assertEqual(
            artifact["state_after"]["outcomes"][0]["metadata"]["request_id"],
            outcome_artifact["request_id"],
        )
        self.assertEqual(
            artifact["state_after"]["outcomes"][0]["metadata"]["protocol_status"],
            outcome_artifact["protocol_status"],
        )
        self.assertEqual(
            artifact["state_after"]["outcomes"][0]["metadata"]["task_status"],
            outcome_artifact["task_status"],
        )
        self.assertIn("output", artifact["state_after"]["outcomes"][0]["metadata"])
        self.assertIn("responder", artifact["state_after"]["outcomes"][0]["metadata"])
        self.assertIn("traceability", artifact["state_after"]["outcomes"][0]["metadata"])
        self.assertEqual(artifact["applied_effects"], ["observations.upsert", "outcomes.upsert"])
        json.dumps(artifact)

    def test_state_feedback_does_not_mutate_input_state(self) -> None:
        result = run_general_read_only_path(now=NOW)
        before_payload = result.state.to_payload()
        outcome_artifact = _outcome_artifact()

        feedback = build_general_state_feedback(
            result.state,
            outcome_artifact,
            now=NOW,
        )

        self.assertEqual(result.state.to_payload(), before_payload)
        self.assertEqual(result.state.outcomes, [])
        self.assertEqual(len(feedback.state_after.outcomes), 1)
        self.assertNotEqual(feedback.state_before.to_payload(), feedback.state_after.to_payload())

    def test_state_feedback_is_idempotent_for_same_outcome_observation(self) -> None:
        result = run_general_read_only_path(now=NOW)
        outcome_artifact = _outcome_artifact()
        first = build_general_state_feedback(
            result.state,
            outcome_artifact,
            now=NOW,
        )
        second = build_general_state_feedback(
            first.state_after,
            outcome_artifact,
            now=NOW,
        )

        self.assertEqual(len(first.state_after.outcomes), 1)
        self.assertEqual(len(second.state_after.outcomes), 1)
        self.assertEqual(second.state_after.outcomes[0].outcome_id, outcome_artifact["outcome_id"])

    def test_state_feedback_validates_outcome_observation_attribution(self) -> None:
        result = run_general_read_only_path(now=NOW)
        outcome_artifact = _outcome_artifact()
        cases = [
            ("decision_id", "decision.other", "metadata.decision_id mismatch"),
            ("trace_ref", "trace.other", "metadata.trace_ref mismatch"),
            ("candidate_id", "candidate.other", "metadata.candidate_id mismatch"),
            ("approval_id", "approval.other", "metadata.approval_id mismatch"),
            ("execution_id", "execution.other", "metadata.execution_id mismatch"),
            ("outcome_id", "outcome.other", "metadata.outcome_id mismatch"),
            ("protocol_status", "error", "metadata.protocol_status mismatch"),
            ("task_status", "failed", "metadata.task_status mismatch"),
        ]

        for field_name, value, error_token in cases:
            with self.subTest(field=field_name):
                artifact = deepcopy(outcome_artifact)
                artifact["outcome_observation"]["metadata"][field_name] = value
                with self.assertRaisesRegex(ValueError, error_token):
                    build_general_state_feedback_artifact(result.state, artifact, now=NOW)

    def test_state_feedback_rejects_missing_required_artifact_fields(self) -> None:
        result = run_general_read_only_path(now=NOW)
        outcome_artifact = _outcome_artifact()
        cases = [
            ("path_type", "path_type is required"),
            ("decision_id", "decision_id is required"),
            ("trace_ref", "trace_ref is required"),
            ("candidate_id", "candidate_id is required"),
            ("execution_id", "execution_id is required"),
            ("outcome_id", "outcome_id is required"),
            ("request_id", "request_id is required"),
            ("protocol_status", "protocol_status is required"),
            ("task_status", "task_status is required"),
        ]

        for field_name, error_token in cases:
            with self.subTest(field=field_name):
                artifact = deepcopy(outcome_artifact)
                artifact.pop(field_name, None)
                with self.assertRaisesRegex(ValueError, error_token):
                    build_general_state_feedback_artifact(result.state, artifact, now=NOW)

    def test_state_feedback_rejects_wrong_outcome_artifact_path_type(self) -> None:
        result = run_general_read_only_path(now=NOW)
        outcome_artifact = _outcome_artifact()
        outcome_artifact["path_type"] = "read_only_general_execution_plan"

        with self.assertRaisesRegex(ValueError, "path_type must be 'read_only_general_outcome_return'"):
            build_general_state_feedback_artifact(result.state, outcome_artifact, now=NOW)

    def test_state_feedback_rejects_outcome_record_mismatch(self) -> None:
        result = run_general_read_only_path(now=NOW)
        outcome_artifact = _outcome_artifact()
        outcome_artifact["outcome_record"]["outcome_id"] = "outcome.other"

        with self.assertRaisesRegex(ValueError, "outcome_record.outcome_id mismatch"):
            build_general_state_feedback_artifact(result.state, outcome_artifact, now=NOW)

    def test_state_feedback_requires_outcome_observation_kind(self) -> None:
        result = run_general_read_only_path(now=NOW)
        outcome_artifact = _outcome_artifact()
        outcome_artifact["outcome_observation"]["kind"] = "signal"

        with self.assertRaisesRegex(ValueError, "kind must be 'outcome'"):
            build_general_state_feedback_artifact(result.state, outcome_artifact, now=NOW)

    def test_state_feedback_preserves_reducible_observation_payload(self) -> None:
        artifact = build_run_demo_general_state_feedback_artifact(now=NOW)
        observation = GenericObservation.from_payload(artifact["outcome_observation"])

        self.assertEqual(observation.kind, "outcome")
        self.assertEqual(observation.metadata["outcome_id"], artifact["outcome_id"])
        self.assertEqual(observation.metadata["execution_id"], artifact["execution_id"])

    def test_run_demo_general_state_feedback_is_read_only(self) -> None:
        stdout = io.StringIO()
        with patch(
            "examples.decision_hub_demo.run_demo.DecisionControlLoop.handle_recommendation",
            side_effect=AssertionError("General state feedback must not enter legacy confirmation loop"),
        ), patch(
            "examples.decision_hub_demo.sdep_executor.create_default_sdep_executor",
            side_effect=AssertionError("General state feedback must not create an SDEP executor"),
        ), patch(
            "spice.executors.sdep.SDEPExecutor.execute",
            side_effect=AssertionError("General state feedback must not execute through SDEP"),
        ), redirect_stdout(stdout):
            run_demo_main(["--general-state-feedback"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["path_type"], "read_only_general_state_feedback")
        self.assertEqual(payload["status"], "state_feedback_applied")
        self.assertTrue(payload["state_updated"])
        self.assertTrue(payload["state_snapshot_updated"])
        self.assertEqual(payload["update_mode"], "read_only_snapshot")
        self.assertFalse(payload["persisted"])
        self.assertFalse(payload["executor_called"])
        self.assertFalse(payload["executed"])
        self.assertIsNone(payload["execution"])


def _outcome_artifact() -> dict[str, object]:
    execution_artifact = build_general_execution_artifact(now=NOW)
    response_payload = build_general_sdep_response_fixture(execution_artifact, now=NOW)
    return build_general_outcome_artifact(
        execution_artifact,
        response_payload,
        now=NOW,
    )


if __name__ == "__main__":
    unittest.main()
