from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from unittest.mock import patch

from spice.decision.compare import render_compare_text
from spice.decision.general import ObservationKind
from spice.protocols.observation import Observation

from examples.decision_hub_demo.general_adapter import (
    build_general_compare_artifact,
    build_general_observations,
    protocol_observation_to_generic,
    run_general_read_only_path,
)
from examples.decision_hub_demo.run_demo import (
    build_compare_artifact,
    build_general_decision_artifact,
    build_legacy_compare_artifact,
    main as run_demo_main,
)


NOW = datetime(2026, 4, 17, 6, 0, tzinfo=timezone.utc)


class DecisionHubGeneralAdapterTests(unittest.TestCase):
    def test_demo_provider_events_normalize_to_general_observations(self) -> None:
        observations = build_general_observations(NOW)
        kinds = {
            item.kind.value if isinstance(item.kind, ObservationKind) else str(item.kind)
            for item in observations
        }

        self.assertIn(ObservationKind.INTENT.value, kinds)
        self.assertIn(ObservationKind.WORK_ITEM.value, kinds)
        self.assertIn(ObservationKind.COMMITMENT.value, kinds)
        self.assertIn(ObservationKind.CAPABILITY.value, kinds)
        self.assertIn(ObservationKind.CONSTRAINT.value, kinds)
        self.assertTrue(
            any(
                item.attributes.get("constraint_kind") == "time_window"
                for item in observations
            )
        )

    def test_general_read_only_path_runs_to_decision_card(self) -> None:
        result = run_general_read_only_path(now=NOW)
        rendered = render_compare_text(result.policy_result.compare_payload, use_bars=False)
        encoded = json.dumps(result.policy_result.to_payload()).lower()

        self.assertEqual(len(result.state.intents), 1)
        self.assertEqual(len(result.state.work_items), 1)
        self.assertEqual(len(result.state.commitments), 1)
        self.assertEqual(len(result.state.capabilities), 1)
        self.assertEqual(len(result.state.constraints), 1)
        self.assertTrue(result.candidates)
        self.assertIn(
            result.policy_result.checkpoint.selected_candidate_id,
            {candidate.candidate_id for candidate in result.candidates},
        )
        self.assertIn("DECISION COMPARISON", rendered)
        self.assertIn("WHY NOT OTHERS", rendered)
        self.assertIn(result.policy_result.checkpoint.selected_candidate_id, rendered)
        self.assertNotIn("sdep", encoded)
        self.assertNotIn("execute.request", encoded)

    def test_general_compare_artifact_is_normalized_and_renderable(self) -> None:
        artifact = build_general_compare_artifact(now=NOW)
        rendered = render_compare_text(artifact, use_bars=False)

        self.assertEqual(
            artifact["decision_id"],
            "decision.general.decision_hub.20260417T060000Z",
        )
        self.assertEqual(
            artifact["trace_ref"],
            "trace.general.decision_hub.20260417T060000Z",
        )
        self.assertTrue(artifact["candidate_decisions"])
        self.assertIn("DECISION COMPARISON", rendered)

    def test_run_demo_build_compare_artifact_defaults_to_general_compare_payload(self) -> None:
        artifact = build_compare_artifact(now=NOW)
        rendered = render_compare_text(artifact, use_bars=False)
        encoded = json.dumps(artifact).lower()

        self.assertIn("decision_id", artifact)
        self.assertIn("trace_ref", artifact)
        self.assertIn("candidate_decisions", artifact)
        self.assertIn("selected_recommendation", artifact)
        self.assertIn("general", artifact["decision_id"])
        self.assertIn("general", artifact["trace_ref"])
        self.assertIn("DECISION COMPARISON", rendered)
        self.assertNotIn("execute.request", encoded)
        self.assertNotIn("cand.delegate_to_executor", encoded)
        self.assertNotIn('"execution_path": "sdep', encoded)

    def test_legacy_compare_artifact_keeps_legacy_semantics(self) -> None:
        artifact = build_legacy_compare_artifact(now=NOW)
        rendered = render_compare_text(artifact, use_bars=False, show_execution=True)

        self.assertEqual(
            artifact["selected_recommendation"]["candidate_id"],
            "cand.delegate_to_executor",
        )
        self.assertEqual(
            artifact["execution_boundary"]["execution_path"],
            "SDEP -> Hermes/Codex",
        )
        self.assertIn("DECISION COMPARISON", rendered)
        self.assertIn("EXECUTION BOUNDARY", rendered)

    def test_general_decision_artifact_envelope_is_json_serializable(self) -> None:
        artifact = build_general_decision_artifact(now=NOW)
        rendered = render_compare_text(artifact["compare_payload"], use_bars=False)

        self.assertEqual(artifact["path_type"], "read_only_general")
        self.assertEqual(artifact["generated_by"], "general_adapter")
        self.assertEqual(
            artifact["decision_id"],
            artifact["compare_payload"]["decision_id"],
        )
        self.assertEqual(
            artifact["trace_ref"],
            artifact["compare_payload"]["trace_ref"],
        )
        self.assertEqual(
            artifact["selected_candidate_id"],
            artifact["compare_payload"]["selected_recommendation"]["candidate_id"],
        )
        self.assertTrue(artifact["observations"])
        self.assertTrue(artifact["general_state_summary"])
        self.assertTrue(artifact["candidates"])
        self.assertIn("DECISION COMPARISON", artifact["rendered_text"])
        self.assertIn("WHY NOT OTHERS", artifact["rendered_text"])
        self.assertEqual(artifact["rendered_text"], rendered)
        json.dumps(artifact)

    def test_run_demo_general_decision_card_is_read_only(self) -> None:
        stdout = io.StringIO()
        with redirect_stdout(stdout):
            run_demo_main(["--general-decision-card", "--no-bars"])

        rendered = stdout.getvalue()

        self.assertIn("DECISION COMPARISON", rendered)
        self.assertIn("WHY NOT OTHERS", rendered)
        self.assertIn("decision.general.decision_hub", rendered)
        self.assertNotIn("scenario_a_confirm", rendered)
        self.assertNotIn("execute.request", rendered.lower())
        self.assertNotIn("sdep", rendered.lower())

    def test_general_decision_card_does_not_call_confirmation_or_sdep(self) -> None:
        stdout = io.StringIO()
        with patch(
            "examples.decision_hub_demo.run_demo.DecisionControlLoop.handle_recommendation",
            side_effect=AssertionError("General decision card must not enter confirmation"),
        ), patch(
            "examples.decision_hub_demo.sdep_executor.create_default_sdep_executor",
            side_effect=AssertionError("General decision card must not create an SDEP executor"),
        ), patch(
            "examples.decision_hub_demo.sdep_executor.execution_request_to_intent",
            side_effect=AssertionError("General decision card must not create ExecutionIntent"),
        ), redirect_stdout(stdout):
            run_demo_main(["--general-decision-card", "--no-bars"])

        self.assertIn("DECISION COMPARISON", stdout.getvalue())

    def test_unknown_protocol_event_stays_record_only(self) -> None:
        unknown = Observation(
            id="obs.demo.unknown",
            timestamp=NOW,
            observation_type="unsupported_demo_event",
            source="manual",
            attributes={"summary": "Unsupported event."},
        )

        observations = protocol_observation_to_generic(unknown)

        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].kind, ObservationKind.UNKNOWN)
        self.assertEqual(observations[0].attributes["summary"], "Unsupported event.")


if __name__ == "__main__":
    unittest.main()
