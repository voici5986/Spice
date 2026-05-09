from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from unittest.mock import patch

from spice.decision.general.candidates import GenericCandidate
from spice.entry.cli import main as cli_main
from spice.llm.candidate_expander import LLMCandidateExpansionResult
from spice.runtime import (
    LocalJsonStore,
    load_workspace_memory_provider,
    refine_decision,
    run_once,
    setup_workspace,
)


NOW = datetime(2026, 4, 29, 6, 0, tzinfo=timezone.utc)


class RuntimeRefineTests(unittest.TestCase):
    def test_refine_adds_manual_candidate_and_saves_updated_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            parent = run_once(
                "Fix the failing test.",
                project_root=tmp_dir,
                now=NOW,
                run_intent_mode="act",
            )

            result = refine_decision(
                "Consider rollback first, then fix forward.",
                project_root=tmp_dir,
                now=NOW,
                run_intent_mode="act",
            )
            store = LocalJsonStore.from_project_root(tmp_dir)
            run_payload = store.load_run(result.artifact["run_id"])

            self.assertEqual(result.artifact["path_type"], "manual_intent_refine")
            self.assertEqual(result.artifact["source"], "manual_refinement")
            self.assertEqual(result.artifact["parent_run_id"], parent.artifact["run_id"])
            memory_records = load_workspace_memory_provider(tmp_dir).query(
                namespace="general.decision",
                limit=-1,
            )
            self.assertEqual(len(memory_records), 2)
            refine_memory = next(
                record
                for record in memory_records
                if record["run_id"] == result.artifact["run_id"]
            )
            self.assertEqual(refine_memory["parent_run_id"], parent.artifact["run_id"])
            self.assertEqual(refine_memory["source"], "manual_refinement")
            self.assertEqual(refine_memory["input"]["text"], "Consider rollback first, then fix forward.")
            self.assertEqual(refine_memory["selected"]["candidate_id"], result.artifact["selected_candidate_id"])
            self.assertEqual(result.artifact["memory_writeback"]["status"], "written")
            state = store.load_state()
            frame = state["world_state"]["domain_state"]["general_decision"]["metadata"][
                "active_decision_frame"
            ]
            self.assertEqual(frame["source"], "manual_refinement")
            self.assertEqual(frame["parent_run_id"], parent.artifact["run_id"])
            self.assertEqual(frame["run_id"], result.artifact["run_id"])
            self.assertEqual(frame["decision_id"], result.artifact["decision_id"])
            self.assertEqual(frame["selected_candidate_id"], result.artifact["selected_candidate_id"])
            self.assertEqual(result.artifact["refinement"]["fallback_used"], True)
            self.assertGreater(result.artifact["refinement"]["added_candidate_count"], 0)
            self.assertIn("compiled_context", result.artifact)
            self.assertIn("context_refs", result.artifact)
            self.assertEqual(
                result.artifact["compiled_context"]["decision_context"]["current_intent"][
                    "parent_run_id"
                ],
                parent.artifact["run_id"],
            )
            self.assertTrue(
                result.artifact["context_refs"]["decision_context_id"].startswith("decision-ctx-")
            )
            self.assertTrue(
                result.artifact["context_refs"]["simulation_context_id"].startswith(
                    "simulation-ctx-"
                )
            )
            self.assertTrue(any(
                candidate.get("metadata", {}).get("source") == "manual_refinement"
                for candidate in result.artifact["candidates"]
            ))
            manual_candidate = next(
                candidate
                for candidate in result.artifact["candidates"]
                if candidate.get("metadata", {}).get("source") == "manual_refinement"
            )
            self.assertEqual(manual_candidate["candidate_kind"], "runtime_action")
            self.assertEqual(
                manual_candidate["metadata"]["candidate_kind"],
                manual_candidate["candidate_kind"],
            )
            self.assertIn("execution_intent", manual_candidate)
            self.assertEqual(
                manual_candidate["metadata"]["execution_intent"]["requested"],
                manual_candidate["execution_intent"]["requested"],
            )
            self.assertIn("SPICE REFINE", result.rendered_text)
            self.assertIn("UPDATED DECISION CARD", result.rendered_text)
            self.assertEqual(run_payload["run_id"], result.artifact["run_id"])
            self.assertTrue(result.decision_path.exists())
            self.assertTrue(result.run_path.exists())

    def test_refine_uses_llm_expansion_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            run_once("Review the project.", project_root=tmp_dir, now=NOW)
            expanded = GenericCandidate(
                candidate_id="candidate.llm.context.refine",
                action_type="context.prepare",
                intent="Prepare rollback context before choosing.",
                target_refs=["intent.manual"],
                requires_confirmation=False,
                side_effect_class="read_only",
                why_available=["LLM proposed a refinement option."],
                availability_status="available",
                metadata={"source": "llm_candidate_expander", "llm_generated": True},
            )
            expansion = LLMCandidateExpansionResult(
                enabled=True,
                status="expanded",
                candidates=[expanded],
                proposed_count=1,
                accepted_count=1,
                model_provider="deterministic",
                model_id="deterministic.v1",
                request_id="refine-llm-test",
            )

            with patch(
                "spice.runtime.refine.expand_candidates_from_runtime_config",
                return_value=expansion,
            ):
                result = refine_decision(
                    "Add rollback context as an option.",
                    project_root=tmp_dir,
                    now=NOW,
                    full_loop_preview=False,
                )

            candidate_ids = [candidate["candidate_id"] for candidate in result.artifact["candidates"]]
            self.assertIn("candidate.llm.context.refine", candidate_ids)
            self.assertFalse(result.artifact["refinement"]["fallback_used"])
            self.assertEqual(result.artifact["llm_candidate_expansion"]["status"], "expanded")
            self.assertNotIn("full_loop_preview", result.artifact)

    def test_refine_requires_prior_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            with self.assertRaisesRegex(ValueError, "No prior run"):
                refine_decision("Try another option.", project_root=tmp_dir, now=NOW)

    def test_refine_does_not_overwrite_repeated_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            run_once("Review the project.", project_root=tmp_dir, now=NOW)

            first = refine_decision("Consider a smaller first step.", project_root=tmp_dir, now=NOW)
            second = refine_decision("Consider a smaller first step.", project_root=tmp_dir, now=NOW)
            store = LocalJsonStore.from_project_root(tmp_dir)

            self.assertNotEqual(first.artifact["run_id"], second.artifact["run_id"])
            self.assertTrue(first.run_path.exists())
            self.assertTrue(second.run_path.exists())
            self.assertGreaterEqual(len(store.list_record_ids("runs")), 3)

    def test_cli_refine_json_outputs_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            run_once("Review the project.", project_root=tmp_dir, now=NOW)
            out = io.StringIO()

            with redirect_stdout(out):
                code = cli_main([
                    "refine",
                    "Consider rollback first.",
                    "--workspace",
                    tmp_dir,
                    "--json",
                    "--decision-only",
                ])

            self.assertEqual(code, 0)
            payload = json.loads(out.getvalue())
            self.assertEqual(payload["path_type"], "manual_intent_refine")
            self.assertEqual(payload["source"], "manual_refinement")
            self.assertEqual(payload["loop_mode"], "decision_only")
            self.assertIn("compare_payload", payload)


if __name__ == "__main__":
    unittest.main()
