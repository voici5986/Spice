from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
from unittest.mock import patch

from spice.decision.general import (
    GeneralDecisionState,
    GenericObservation,
    reduce_generic_observations,
)
from spice.protocols.sdep import SDEPExecuteResponse

from examples.decision_hub_demo.general_outcome import (
    build_general_outcome_artifact,
    build_general_outcome_return,
    build_general_sdep_response_fixture,
)
from examples.decision_hub_demo.run_demo import (
    build_general_execution_artifact,
    build_general_outcome_artifact as build_run_demo_general_outcome_artifact,
    main as run_demo_main,
)


NOW = datetime(2026, 4, 17, 6, 0, tzinfo=timezone.utc)


class DecisionHubGeneralOutcomeTests(unittest.TestCase):
    def test_sdep_response_returns_read_only_outcome_artifact(self) -> None:
        execution_artifact = build_general_execution_artifact(now=NOW)
        response_payload = build_general_sdep_response_fixture(
            execution_artifact,
            now=NOW,
        )

        artifact = build_general_outcome_artifact(
            execution_artifact,
            response_payload,
            now=NOW,
        )

        self.assertEqual(artifact["path_type"], "read_only_general_outcome_return")
        self.assertEqual(artifact["generated_by"], "decision_hub_demo.general_outcome_adapter")
        self.assertEqual(artifact["status"], "outcome_observed")
        self.assertEqual(artifact["decision_id"], execution_artifact["decision_id"])
        self.assertEqual(artifact["trace_ref"], execution_artifact["trace_ref"])
        self.assertEqual(artifact["candidate_id"], execution_artifact["candidate_id"])
        self.assertEqual(artifact["approval_id"], execution_artifact["approval_id"])
        self.assertEqual(artifact["execution_id"], execution_artifact["execution_id"])
        self.assertEqual(artifact["outcome_id"], artifact["outcome_record"]["outcome_id"])
        self.assertEqual(artifact["protocol_status"], "success")
        self.assertEqual(artifact["task_status"], "success")
        self.assertTrue(artifact["response_processed"])
        self.assertFalse(artifact["executor_called"])
        self.assertFalse(artifact["executed"])
        self.assertIsNone(artifact["execution"])
        self.assertFalse(artifact["state_updated"])

        outcome = artifact["outcome_record"]
        observation = artifact["outcome_observation"]
        self.assertEqual(outcome["decision_id"], execution_artifact["decision_id"])
        self.assertEqual(outcome["trace_ref"], execution_artifact["trace_ref"])
        self.assertEqual(outcome["candidate_id"], execution_artifact["candidate_id"])
        self.assertEqual(outcome["execution_ref"], execution_artifact["execution_id"])
        self.assertEqual(outcome["protocol_status"], "success")
        self.assertEqual(outcome["task_status"], "success")
        self.assertEqual(outcome["metadata"]["approval_id"], execution_artifact["approval_id"])
        self.assertEqual(outcome["metadata"]["execution_id"], execution_artifact["execution_id"])
        self.assertEqual(outcome["metadata"]["request_id"], execution_artifact["sdep_request"]["request_id"])
        self.assertEqual(observation["kind"], "outcome")
        self.assertEqual(observation["attributes"]["execution_ref"], execution_artifact["execution_id"])
        self.assertEqual(observation["metadata"]["decision_id"], execution_artifact["decision_id"])
        self.assertEqual(observation["metadata"]["trace_ref"], execution_artifact["trace_ref"])
        self.assertEqual(observation["metadata"]["candidate_id"], execution_artifact["candidate_id"])
        self.assertEqual(observation["metadata"]["approval_id"], execution_artifact["approval_id"])
        self.assertEqual(observation["metadata"]["execution_id"], execution_artifact["execution_id"])
        self.assertEqual(observation["metadata"]["outcome_id"], artifact["outcome_id"])
        self.assertEqual(
            observation["metadata"]["request_id"],
            execution_artifact["sdep_request"]["request_id"],
        )
        json.dumps(artifact)

    def test_protocol_success_task_failed_status_split_is_preserved(self) -> None:
        execution_artifact = build_general_execution_artifact(now=NOW)
        response_payload = build_general_sdep_response_fixture(
            execution_artifact,
            now=NOW,
            response_status="success",
            task_status="failed",
            output={
                "summary": "Executor reached the target but the task failed.",
                "state_delta": {"task_status": "failed"},
            },
        )

        artifact = build_general_outcome_artifact(
            execution_artifact,
            response_payload,
            now=NOW,
        )

        self.assertEqual(artifact["protocol_status"], "success")
        self.assertEqual(artifact["task_status"], "failed")
        self.assertEqual(artifact["outcome_record"]["protocol_status"], "success")
        self.assertEqual(artifact["outcome_record"]["task_status"], "failed")
        self.assertEqual(artifact["outcome_record"]["state_delta"], {"task_status": "failed"})

    def test_protocol_error_task_failed_status_split_is_preserved(self) -> None:
        execution_artifact = build_general_execution_artifact(now=NOW)
        response_payload = build_general_sdep_response_fixture(
            execution_artifact,
            now=NOW,
            response_status="error",
            task_status="failed",
            output={
                "summary": "Wrapper failed before the task completed.",
                "state_delta": {"task_status": "failed", "protocol_status": "error"},
            },
        )

        artifact = build_general_outcome_artifact(
            execution_artifact,
            response_payload,
            now=NOW,
        )

        self.assertEqual(artifact["protocol_status"], "error")
        self.assertEqual(artifact["task_status"], "failed")
        self.assertEqual(artifact["outcome_record"]["protocol_status"], "error")
        self.assertEqual(artifact["outcome_record"]["task_status"], "failed")
        self.assertEqual(artifact["outcome_observation"]["metadata"]["protocol_status"], "error")
        self.assertEqual(artifact["outcome_observation"]["metadata"]["task_status"], "failed")

    def test_outcome_id_is_stable_and_includes_protocol_and_task_status(self) -> None:
        execution_artifact = build_general_execution_artifact(now=NOW)
        success_response = build_general_sdep_response_fixture(
            execution_artifact,
            now=NOW,
            response_status="success",
            task_status="success",
        )
        same_success = build_general_sdep_response_fixture(
            execution_artifact,
            now=NOW,
            response_status="success",
            task_status="success",
        )
        task_failed = build_general_sdep_response_fixture(
            execution_artifact,
            now=NOW,
            response_status="success",
            task_status="failed",
        )
        protocol_error = build_general_sdep_response_fixture(
            execution_artifact,
            now=NOW,
            response_status="error",
            task_status="success",
        )

        first = build_general_outcome_artifact(execution_artifact, success_response, now=NOW)
        second = build_general_outcome_artifact(execution_artifact, same_success, now=NOW)
        failed = build_general_outcome_artifact(execution_artifact, task_failed, now=NOW)
        errored = build_general_outcome_artifact(execution_artifact, protocol_error, now=NOW)

        self.assertEqual(first["outcome_id"], second["outcome_id"])
        self.assertNotEqual(first["outcome_id"], failed["outcome_id"])
        self.assertNotEqual(first["outcome_id"], errored["outcome_id"])

    def test_response_payload_round_trips_through_sdep_model(self) -> None:
        execution_artifact = build_general_execution_artifact(now=NOW)
        response_payload = build_general_sdep_response_fixture(execution_artifact, now=NOW)

        response = SDEPExecuteResponse.from_dict(response_payload)
        wire_payload = response.to_dict()

        self.assertEqual(wire_payload["message_type"], "execute.response")
        self.assertEqual(wire_payload["request_id"], execution_artifact["sdep_request"]["request_id"])
        self.assertEqual(wire_payload["outcome"]["execution_id"], execution_artifact["execution_id"])
        self.assertEqual(wire_payload["traceability"]["spice_decision_id"], execution_artifact["decision_id"])
        self.assertEqual(wire_payload["traceability"]["trace_ref"], execution_artifact["trace_ref"])
        self.assertEqual(wire_payload["traceability"]["candidate_id"], execution_artifact["candidate_id"])
        self.assertEqual(wire_payload["traceability"]["approval_id"], execution_artifact["approval_id"])

    def test_response_attribution_mismatch_raises_value_error(self) -> None:
        execution_artifact = build_general_execution_artifact(now=NOW)
        base_response = build_general_sdep_response_fixture(execution_artifact, now=NOW)
        cases = [
            ("request_id", "sdep-req.other", "request_id mismatch"),
            ("outcome.execution_id", "exec.other", "execution_id mismatch"),
            ("traceability.execution_id", "exec.other", "traceability.execution_id mismatch"),
            ("traceability.spice_decision_id", "decision.other", "traceability.spice_decision_id mismatch"),
            ("traceability.trace_ref", "trace.other", "traceability.trace_ref mismatch"),
            ("traceability.candidate_id", "candidate.other", "traceability.candidate_id mismatch"),
            ("traceability.approval_id", "approval.other", "traceability.approval_id mismatch"),
        ]

        for field_name, value, error_token in cases:
            with self.subTest(field=field_name):
                response = deepcopy(base_response)
                _set_nested(response, field_name, value)
                with self.assertRaisesRegex(ValueError, error_token):
                    build_general_outcome_return(execution_artifact, response, now=NOW)

    def test_missing_response_attribution_field_raises_value_error(self) -> None:
        execution_artifact = build_general_execution_artifact(now=NOW)
        base_response = build_general_sdep_response_fixture(execution_artifact, now=NOW)
        cases = [
            ("traceability.execution_id", "traceability.execution_id mismatch"),
            ("traceability.spice_decision_id", "traceability.spice_decision_id mismatch"),
            ("traceability.trace_ref", "traceability.trace_ref mismatch"),
            ("traceability.candidate_id", "traceability.candidate_id mismatch"),
            ("traceability.approval_id", "traceability.approval_id mismatch"),
        ]

        for field_name, error_token in cases:
            with self.subTest(field=field_name):
                response = deepcopy(base_response)
                _delete_nested(response, field_name)
                with self.assertRaisesRegex(ValueError, error_token):
                    build_general_outcome_return(execution_artifact, response, now=NOW)

    def test_outcome_observation_can_be_reduced_later_without_adapter_state_write(self) -> None:
        execution_artifact = build_general_execution_artifact(now=NOW)
        response_payload = build_general_sdep_response_fixture(execution_artifact, now=NOW)
        artifact = build_general_outcome_artifact(
            execution_artifact,
            response_payload,
            now=NOW,
        )

        original = GeneralDecisionState(state_id="world.general.outcome")
        reduced = reduce_generic_observations(
            original,
            [
                # The adapter returns a stable GenericObservation payload, but does not
                # apply it to state by itself.
                GenericObservation.from_payload(artifact["outcome_return"]["outcome_observation"]),
            ],
        )

        self.assertEqual(original.outcomes, [])
        self.assertEqual(len(reduced.outcomes), 1)
        self.assertEqual(reduced.outcomes[0].decision_id, execution_artifact["decision_id"])
        self.assertEqual(reduced.outcomes[0].execution_ref, execution_artifact["execution_id"])
        self.assertEqual(reduced.outcomes[0].metadata["approval_id"], execution_artifact["approval_id"])
        self.assertEqual(
            reduced.outcomes[0].metadata["request_id"],
            execution_artifact["sdep_request"]["request_id"],
        )
        self.assertEqual(reduced.outcomes[0].metadata["execution_id"], execution_artifact["execution_id"])
        self.assertEqual(reduced.outcomes[0].metadata["outcome_id"], artifact["outcome_id"])
        self.assertIn("responder", reduced.outcomes[0].metadata)

    def test_run_demo_general_outcome_return_is_read_only(self) -> None:
        stdout = io.StringIO()
        with patch(
            "examples.decision_hub_demo.run_demo.DecisionControlLoop.handle_recommendation",
            side_effect=AssertionError("General outcome return must not enter legacy confirmation loop"),
        ), patch(
            "examples.decision_hub_demo.sdep_executor.create_default_sdep_executor",
            side_effect=AssertionError("General outcome return must not create an SDEP executor"),
        ), patch(
            "spice.executors.sdep.SDEPExecutor.execute",
            side_effect=AssertionError("General outcome return must not execute through SDEP"),
        ), redirect_stdout(stdout):
            run_demo_main(["--general-outcome-return"])

        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["path_type"], "read_only_general_outcome_return")
        self.assertEqual(payload["status"], "outcome_observed")
        self.assertEqual(payload["protocol_status"], "success")
        self.assertEqual(payload["task_status"], "success")
        self.assertFalse(payload["executor_called"])
        self.assertFalse(payload["executed"])
        self.assertIsNone(payload["execution"])
        self.assertFalse(payload["state_updated"])

    def test_run_demo_general_outcome_builder_uses_fixture_response(self) -> None:
        artifact = build_run_demo_general_outcome_artifact(now=NOW)

        self.assertEqual(artifact["path_type"], "read_only_general_outcome_return")
        self.assertTrue(artifact["sdep_response"]["metadata"]["fixture"])
        self.assertEqual(artifact["outcome_record"]["execution_ref"], artifact["execution_id"])
        self.assertEqual(artifact["outcome_observation"]["kind"], "outcome")


def _set_nested(payload: dict, field_name: str, value: object) -> None:
    current = payload
    parts = field_name.split(".")
    for part in parts[:-1]:
        current = current[part]
    current[parts[-1]] = value


def _delete_nested(payload: dict, field_name: str) -> None:
    current = payload
    parts = field_name.split(".")
    for part in parts[:-1]:
        current = current[part]
    current.pop(parts[-1], None)


if __name__ == "__main__":
    unittest.main()
