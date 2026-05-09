from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone

from spice.entry.cli import main as spice_cli_main
from spice.runtime import (
    LocalJsonStore,
    approve_approval,
    execute_sdep_subprocess_approval,
    load_workspace_memory_provider,
    reject_approval,
    run_once,
    setup_workspace,
)


NOW = datetime(2026, 4, 29, 6, 0, tzinfo=timezone.utc)
ECHO_COMMAND = [sys.executable, "-m", "spice.runtime.sdep_echo_executor"]


class RuntimeSDEPSubprocessExecutorTests(unittest.TestCase):
    def test_approved_approval_executes_through_echo_subprocess(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, approval_id, run_id = _approved_handoff(tmp_dir)

            result = execute_sdep_subprocess_approval(
                approval_id,
                project_root=tmp_dir,
                command=ECHO_COMMAND,
                now=NOW,
            )

            artifact = result.artifact
            outcome = store.load_outcome(artifact["outcome_id"])
            run_payload = store.load_run(run_id)
            session = store.load_session("session.default")
            state = store.load_state()
            general = state["world_state"]["domain_state"]["general_decision"]
            self.assertEqual(artifact["executor_provider"], "sdep_subprocess")
            self.assertTrue(artifact["sdep_request_sent"])
            self.assertTrue(artifact["executor_called"])
            self.assertTrue(artifact["transport_executor_called"])
            self.assertFalse(artifact["real_executor_called"])
            self.assertTrue(artifact["executed"])
            self.assertEqual(artifact["protocol_status"], "success")
            self.assertEqual(artifact["task_status"], "success")
            self.assertEqual(outcome["outcome_id"], artifact["outcome_id"])
            self.assertEqual(run_payload["outcome_id"], artifact["outcome_id"])
            self.assertEqual(run_payload["executor_provider"], "sdep_subprocess")
            self.assertEqual(session["metadata"]["last_outcome_id"], artifact["outcome_id"])
            self.assertEqual(session["metadata"]["last_executor_provider"], "sdep_subprocess")
            self.assertEqual(len(general["outcomes"]), 1)
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
            self.assertEqual(memory["executor"]["provider"], "sdep_subprocess")
            self.assertTrue(memory["executor"]["sdep_request_sent"])
            self.assertEqual(memory["executor"]["command"], " ".join(ECHO_COMMAND))
            self.assertEqual(memory["execution"]["task_status"], "success")
            self.assertTrue(memory["execution"]["success"])
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
            self.assertEqual(
                summary_records[-1]["execution_outcomes"][0]["executor"],
                "sdep_subprocess",
            )
            self.assertIn("SPICE SDEP SUBPROCESS EXECUTION", result.rendered_text)

    def test_pending_approval_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            result = run_once(
                "Fix the failing test.",
                project_root=tmp_dir,
                now=NOW,
                run_intent_mode="act",
            )

            with self.assertRaises(ValueError):
                execute_sdep_subprocess_approval(
                    result.artifact["approval_id"],
                    project_root=tmp_dir,
                    command=ECHO_COMMAND,
                    now=NOW,
                )

    def test_rejected_approval_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, approval_id, _run_id = _pending_handoff(tmp_dir)
            reject_approval(store, approval_id, now=NOW)

            with self.assertRaises(ValueError):
                execute_sdep_subprocess_approval(
                    approval_id,
                    project_root=tmp_dir,
                    command=ECHO_COMMAND,
                    now=NOW,
                )

    def test_invalid_subprocess_json_does_not_write_outcome_or_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, approval_id, _run_id = _approved_handoff(tmp_dir)
            state_before = store.load_state()

            with self.assertRaises(ValueError):
                execute_sdep_subprocess_approval(
                    approval_id,
                    project_root=tmp_dir,
                    command=[sys.executable, "-c", "print('not-json')"],
                    now=NOW,
                )

            self.assertEqual(store.list_record_ids("outcomes"), [])
            self.assertEqual(store.load_state(), state_before)

    def test_attribution_mismatch_does_not_write_outcome_or_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, approval_id, _run_id = _approved_handoff(tmp_dir)
            state_before = store.load_state()
            code = (
                "import json, sys; "
                "req=json.loads(sys.stdin.read()); "
                "tr=dict(req.get('traceability') or {}); "
                "tr['execution_id']='wrong.execution'; "
                "resp={"
                "'protocol':'sdep','sdep_version':'0.1','message_type':'execute.response',"
                "'message_id':'sdep-msg.mismatch','request_id':req['request_id'],"
                "'timestamp':req['timestamp'],"
                "'responder':{'id':'bad','name':'Bad','version':'0.1','implementation':'fixture','role':'executor'},"
                "'status':'success',"
                "'outcome':{'execution_id':'wrong.execution','status':'success','outcome_type':'observation','output':{},'artifacts':[],'metrics':{},'metadata':{}},"
                "'traceability':tr,'metadata':{}}; "
                "print(json.dumps(resp))"
            )

            with self.assertRaises(ValueError):
                execute_sdep_subprocess_approval(
                    approval_id,
                    project_root=tmp_dir,
                    command=[sys.executable, "-c", code],
                    now=NOW,
                )

            self.assertEqual(store.list_record_ids("outcomes"), [])
            self.assertEqual(store.load_state(), state_before)

    def test_timeout_does_not_write_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, approval_id, _run_id = _approved_handoff(tmp_dir)
            state_before = store.load_state()

            with self.assertRaises(TimeoutError):
                execute_sdep_subprocess_approval(
                    approval_id,
                    project_root=tmp_dir,
                    command=[sys.executable, "-c", "import time; time.sleep(2)"],
                    timeout_seconds=1,
                    now=NOW,
                )

            self.assertEqual(store.list_record_ids("outcomes"), [])
            self.assertEqual(store.load_state(), state_before)

    def test_cli_execute_sdep_outputs_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, approval_id, _run_id = _approved_handoff(tmp_dir)
            stdout = io.StringIO()
            command = f"{sys.executable} -m spice.runtime.sdep_echo_executor"

            with redirect_stdout(stdout):
                exit_code = spice_cli_main(
                    [
                        "execute",
                        "sdep",
                        approval_id,
                        "--workspace",
                        tmp_dir,
                        "--command",
                        command,
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            artifact = json.loads(stdout.getvalue())
            self.assertEqual(artifact["executor_provider"], "sdep_subprocess")
            self.assertTrue(artifact["sdep_request_sent"])
            self.assertEqual(len(store.list_record_ids("outcomes")), 1)

    def test_cli_execute_sdep_missing_workspace_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                exit_code = spice_cli_main(
                    [
                        "execute",
                        "sdep",
                        "approval.missing",
                        "--workspace",
                        tmp_dir,
                        "--command",
                        f"{sys.executable} -m spice.runtime.sdep_echo_executor",
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("Run `spice setup` first", stderr.getvalue())

    def test_cli_execute_sdep_missing_approval_fails_cleanly(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            stderr = io.StringIO()

            with redirect_stderr(stderr):
                exit_code = spice_cli_main(
                    [
                        "execute",
                        "sdep",
                        "approval.missing",
                        "--workspace",
                        tmp_dir,
                        "--command",
                        f"{sys.executable} -m spice.runtime.sdep_echo_executor",
                    ]
                )

            self.assertEqual(exit_code, 1)
            self.assertIn("approval.missing", stderr.getvalue())


def _pending_handoff(tmp_dir: str) -> tuple[LocalJsonStore, str, str]:
    setup_workspace(project_root=tmp_dir)
    result = run_once(
        "Fix the failing test.",
        project_root=tmp_dir,
        now=NOW,
        run_intent_mode="act",
    )
    store = LocalJsonStore.from_project_root(tmp_dir)
    return store, result.artifact["approval_id"], result.artifact["run_id"]


def _approved_handoff(tmp_dir: str) -> tuple[LocalJsonStore, str, str]:
    store, approval_id, run_id = _pending_handoff(tmp_dir)
    approve_approval(store, approval_id, now=NOW)
    return store, approval_id, run_id


if __name__ == "__main__":
    unittest.main()
