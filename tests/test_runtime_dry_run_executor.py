from __future__ import annotations

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone

from spice.decision.general.approval import Approval
from spice.entry.cli import main as spice_cli_main
from spice.protocols.sdep import SDEPExecuteResponse
from spice.runtime import (
    LocalJsonStore,
    execute_dry_run_approval,
    load_workspace_memory_provider,
    run_once,
    setup_workspace,
)


NOW = datetime(2026, 4, 29, 6, 0, tzinfo=timezone.utc)


class RuntimeDryRunExecutorTests(unittest.TestCase):
    def test_approved_approval_generates_outcome_and_updates_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, approval_id, run_id = _setup_approved_runtime_handoff(tmp_dir)

            result = execute_dry_run_approval(approval_id, project_root=tmp_dir, now=NOW)
            artifact = result.artifact
            state = store.load_state()
            run = store.load_run(run_id)
            outcome = store.load_outcome(artifact["outcome_id"])
            response = SDEPExecuteResponse.from_dict(artifact["sdep_response"])

            self.assertEqual(artifact["path_type"], "runtime_dry_run_execution")
            self.assertTrue(artifact["dry_run"])
            self.assertTrue(artifact["dry_run_executor_called"])
            self.assertFalse(artifact["real_executor_called"])
            self.assertFalse(artifact["executor_called"])
            self.assertFalse(artifact["sdep_request_sent"])
            self.assertFalse(artifact["executed"])
            self.assertTrue(artifact["state_updated"])
            self.assertTrue(artifact["persisted"])
            self.assertEqual(response.message_type, "execute.response")
            self.assertEqual(response.status, "success")
            self.assertEqual(response.outcome.status, "success")
            self.assertEqual(outcome["outcome_id"], artifact["outcome_id"])
            self.assertEqual(run["dry_run_execution"]["outcome_id"], artifact["outcome_id"])
            general = state["world_state"]["domain_state"]["general_decision"]
            self.assertEqual(len(general["outcomes"]), 1)
            reduced = general["outcomes"][0]
            self.assertEqual(reduced["outcome_id"], artifact["outcome_id"])
            self.assertEqual(reduced["metadata"]["request_id"], artifact["request_id"])
            self.assertEqual(reduced["metadata"]["approval_id"], approval_id)
            self.assertEqual(reduced["protocol_status"], "success")
            self.assertEqual(reduced["task_status"], "success")
            memory_records = load_workspace_memory_provider(tmp_dir).query(
                namespace="general.reflection",
                limit=-1,
            )
            self.assertEqual(len(memory_records), 1)
            memory = memory_records[0]
            self.assertEqual(artifact["memory_writeback"]["status"], "written")
            self.assertEqual(artifact["memory_writeback"]["namespace"], "general.reflection")
            self.assertEqual(memory["run_id"], run_id)
            self.assertEqual(memory["approval_id"], approval_id)
            self.assertEqual(memory["candidate_id"], artifact["candidate_id"])
            self.assertEqual(memory["executor"]["provider"], "dry_run")
            self.assertFalse(memory["executor"]["real_executor_called"])
            self.assertEqual(memory["execution"]["task_status"], "success")
            self.assertTrue(memory["execution"]["success"])
            self.assertEqual(memory["state_delta_summary"]["task_status"], "success")
            self.assertEqual(
                memory["state_delta_summary"]["state_after_ref"],
                artifact["state_after_ref"],
            )
            summary_records = load_workspace_memory_provider(tmp_dir).query(
                namespace="general.session_summary",
                limit=-1,
            )
            self.assertEqual(len(summary_records), 2)
            self.assertEqual(summary_records[-1]["execution_outcomes"][0]["task_status"], "success")
            self.assertEqual(summary_records[-1]["execution_outcomes"][0]["executor"], "dry_run")
            self.assertFalse(
                [
                    thread
                    for thread in summary_records[-1]["open_threads"]
                    if thread.get("kind") == "approval"
                ]
            )
            self.assertIn("SPICE DRY-RUN EXECUTION", result.rendered_text)
            json.dumps(artifact)

    def test_pending_or_rejected_approval_cannot_execute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _, pending_id, _ = _setup_approved_runtime_handoff(
                tmp_dir,
                status="pending",
                execution_allowed=False,
            )
            with self.assertRaisesRegex(ValueError, "must be approved"):
                execute_dry_run_approval(pending_id, project_root=tmp_dir, now=NOW)

        with tempfile.TemporaryDirectory() as tmp_dir:
            _, rejected_id, _ = _setup_approved_runtime_handoff(
                tmp_dir,
                status="rejected",
                execution_allowed=False,
            )
            with self.assertRaisesRegex(ValueError, "must be approved"):
                execute_dry_run_approval(rejected_id, project_root=tmp_dir, now=NOW)

    def test_missing_associated_run_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            store = LocalJsonStore.from_project_root(tmp_dir)
            approval = Approval(
                approval_id="approval.missing.run",
                decision_id="decision.test",
                candidate_id="candidate.test",
                status="approved",
                execution_allowed=True,
            )
            store.save_approval(approval.approval_id, approval.to_payload())

            with self.assertRaisesRegex(ValueError, "No run artifact found"):
                execute_dry_run_approval(approval.approval_id, project_root=tmp_dir, now=NOW)

    def test_cli_execute_dry_run_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            _, approval_id, _ = _setup_approved_runtime_handoff(tmp_dir)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = spice_cli_main(
                    [
                        "execute",
                        "dry-run",
                        approval_id,
                        "--workspace",
                        tmp_dir,
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            artifact = json.loads(stdout.getvalue())
            self.assertEqual(artifact["path_type"], "runtime_dry_run_execution")
            self.assertTrue(artifact["dry_run_executor_called"])
            self.assertFalse(artifact["real_executor_called"])
            self.assertFalse(artifact["sdep_request_sent"])
            self.assertTrue(artifact["state_updated"])


def _setup_approved_runtime_handoff(
    tmp_dir: str,
    *,
    status: str = "approved",
    execution_allowed: bool = True,
) -> tuple[LocalJsonStore, str, str]:
    setup_workspace(project_root=tmp_dir)
    result = run_once(
        "Review the project and pick the safest next action.",
        project_root=tmp_dir,
        now=NOW,
        run_intent_mode="act",
    )
    store = LocalJsonStore.from_project_root(tmp_dir)
    run = store.load_run(result.artifact["run_id"])
    preview = run["full_loop_preview"]
    approval_id = run["approval_id"]

    approval = Approval(
        approval_id=approval_id,
        decision_id=run["decision_id"],
        candidate_id=preview["selected_candidate_id"],
        status=status,
        mode="confirm_before_execution",
        requested_at="2026-04-29T06:00:00+00:00",
        resolved_at="2026-04-29T06:00:00+00:00" if status == "approved" else "",
        actor="test",
        prompt="Approve dry-run handoff?",
        response=status,
        execution_allowed=execution_allowed,
        metadata={"trace_ref": run["trace_ref"]},
    )
    run["approval"] = approval.to_payload()
    store.save_approval(approval.approval_id, approval.to_payload())
    store.save_run(run["run_id"], run)
    session = store.load_session("session.default")
    session["approval_ids"] = [approval.approval_id]
    session["pending_approval_ids"] = [] if status == "approved" else [approval.approval_id]
    store.save_session("session.default", session)
    return store, approval.approval_id, run["run_id"]


if __name__ == "__main__":
    unittest.main()
