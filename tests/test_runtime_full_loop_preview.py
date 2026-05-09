from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone

from spice.protocols.sdep import SDEPExecuteRequest, SDEPExecuteResponse
from spice.runtime import LocalJsonStore, run_once, setup_workspace


NOW = datetime(2026, 4, 29, 6, 0, tzinfo=timezone.utc)


class RuntimeFullLoopPreviewTests(unittest.TestCase):
    def test_run_once_full_loop_preview_adds_skill_context_and_sdep_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            result = run_once(
                "Review this repo and suggest the safest next action",
                project_root=tmp_dir,
                now=NOW,
                full_loop_preview=True,
                use_bars=False,
            )
            artifact = result.artifact
            preview = artifact["full_loop_preview"]

            self.assertEqual(artifact["loop_mode"], "full_loop_preview")
            self.assertEqual(preview["path_type"], "runtime_full_loop_preview")
            self.assertEqual(preview["loop_status"], "completed_read_only")
            self.assertFalse(preview["executor_called"])
            self.assertFalse(preview["sdep_request_sent"])
            self.assertFalse(preview["executed"])
            self.assertFalse(preview["persisted"])
            self.assertTrue(preview["state_snapshot_updated"])
            self.assertEqual(preview["update_mode"], "read_only_snapshot")
            self.assertEqual(artifact["skill_id"], preview["skill_id"])
            self.assertEqual(artifact["executor_id"], preview["executor_id"])
            self.assertEqual(artifact["context_pack_id"], preview["context_pack_id"])
            self.assertEqual(artifact["execution_id"], preview["execution_id"])
            self.assertEqual(artifact["request_id"], preview["request_id"])
            self.assertEqual(artifact["outcome_id"], preview["outcome_id"])
            self.assertIn("resolved_skill", preview)
            self.assertIn("context_pack", preview)
            self.assertIn("sdep_request", preview)
            self.assertIn("sdep_response_fixture", preview)
            json.dumps(artifact)

    def test_full_loop_preview_keeps_active_state_decision_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            result = run_once(
                "Review this repo and suggest the safest next action",
                project_root=tmp_dir,
                now=NOW,
                full_loop_preview=True,
            )
            store = LocalJsonStore.from_project_root(tmp_dir)
            state = store.load_state()
            general = state["world_state"]["domain_state"]["general_decision"]

            self.assertEqual(len(general["outcomes"]), 0)
            self.assertEqual(result.artifact["full_loop_preview"]["state_feedback"]["state_after_summary"]["outcome_count"], 1)
            self.assertTrue(result.artifact["persisted"])
            self.assertFalse(result.artifact["full_loop_preview"]["persisted"])

    def test_full_loop_preview_sdep_payloads_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            result = run_once(
                "Review this repo and suggest the safest next action",
                project_root=tmp_dir,
                now=NOW,
                full_loop_preview=True,
            )
            preview = result.artifact["full_loop_preview"]
            request = SDEPExecuteRequest.from_dict(preview["sdep_request"])
            response = SDEPExecuteResponse.from_dict(preview["sdep_response_fixture"])

            self.assertEqual(request.message_type, "execute.request")
            self.assertEqual(response.message_type, "execute.response")
            self.assertEqual(request.request_id, preview["request_id"])
            self.assertEqual(response.request_id, preview["request_id"])
            self.assertEqual(request.execution.input["context_pack"]["context_pack_id"], preview["context_pack_id"])
            self.assertEqual(request.execution.input["skill_hint"]["skill_id"], preview["skill_id"])
            self.assertTrue(request.metadata["planning_only"])

    def test_full_loop_preview_human_output_contains_read_only_markers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            result = run_once(
                "Review this repo and suggest the safest next action",
                project_root=tmp_dir,
                now=NOW,
                full_loop_preview=True,
                use_bars=False,
            )
            rendered = result.rendered_text

            self.assertIn("SPICE DECISION LOOP", rendered)
            self.assertIn("SKILL RESOLUTION", rendered)
            self.assertIn("CONTEXT PACK", rendered)
            self.assertIn("EXECUTION BOUNDARY", rendered)
            self.assertIn("sdep_request_sent: false", rendered)
            self.assertIn("executor_called: false", rendered)
            self.assertIn("full_loop_feedback_persisted: false", rendered)
            self.assertNotIn("executed successfully", rendered)


if __name__ == "__main__":
    unittest.main()
