from __future__ import annotations

import io
import json
import shlex
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone

from spice.entry.cli import main as spice_cli_main
from spice.runtime import (
    LocalJsonStore,
    approve_approval,
    execute_claude_code_approval,
    run_once,
    setup_workspace,
)
from spice.runtime.claude_code_executor import execute_claude_code_sdep_request
from spice.runtime.workspace import update_workspace_config


NOW = datetime(2026, 4, 29, 6, 0, tzinfo=timezone.utc)


class RuntimeClaudeCodeExecutorTests(unittest.TestCase):
    def test_claude_code_sdep_endpoint_returns_valid_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, _approval_id, run_id = _approved_handoff(tmp_dir)
            request = store.load_run(run_id)["full_loop_preview"]["sdep_request"]

            response = execute_claude_code_sdep_request(
                request,
                command=_fake_claude_command("claude code fixture completed"),
            )

            self.assertEqual(response["message_type"], "execute.response")
            self.assertEqual(response["request_id"], request["request_id"])
            self.assertEqual(response["status"], "success")
            self.assertEqual(response["outcome"]["status"], "success")
            self.assertEqual(response["responder"]["id"], "claude_code")
            self.assertEqual(response["metadata"]["executor_provider"], "claude_code")
            self.assertEqual(
                response["traceability"]["approval_id"],
                request["traceability"]["approval_id"],
            )

    def test_approved_approval_executes_through_claude_code_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, approval_id, run_id = _approved_handoff(tmp_dir)

            result = execute_claude_code_approval(
                approval_id,
                project_root=tmp_dir,
                command=_fake_claude_command("claude code fixture completed"),
                now=NOW,
            )

            artifact = result.artifact
            run_payload = store.load_run(run_id)
            outcome = store.load_outcome(artifact["outcome_id"])
            state = store.load_state()
            general = state["world_state"]["domain_state"]["general_decision"]
            self.assertEqual(artifact["executor_provider"], "claude_code")
            self.assertTrue(artifact["sdep_request_sent"])
            self.assertTrue(artifact["executor_called"])
            self.assertTrue(artifact["transport_executor_called"])
            self.assertTrue(artifact["real_executor_called"])
            self.assertEqual(artifact["protocol_status"], "success")
            self.assertEqual(artifact["task_status"], "success")
            self.assertEqual(outcome["executor_provider"], "claude_code")
            self.assertEqual(run_payload["executor_provider"], "claude_code")
            self.assertIn("claude_code_execution", run_payload)
            self.assertEqual(len(general["outcomes"]), 1)

    def test_default_execute_dispatch_uses_claude_code_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, approval_id, _run_id = _approved_handoff(tmp_dir)
            update_workspace_config(tmp_dir, "executor", "claude_code")
            update_workspace_config(
                tmp_dir,
                "executor_command",
                shlex.join(_fake_claude_command("claude code default dispatch")),
            )
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = spice_cli_main(
                    [
                        "execute",
                        approval_id,
                        "--workspace",
                        tmp_dir,
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            artifact = json.loads(stdout.getvalue())
            self.assertEqual(artifact["executor_provider"], "claude_code")
            self.assertEqual(len(store.list_record_ids("outcomes")), 1)

    def test_cli_execute_claude_code_override_outputs_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, approval_id, _run_id = _approved_handoff(tmp_dir)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = spice_cli_main(
                    [
                        "execute",
                        "claude-code",
                        approval_id,
                        "--workspace",
                        tmp_dir,
                        "--command",
                        shlex.join(_fake_claude_command("claude code explicit override")),
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            artifact = json.loads(stdout.getvalue())
            self.assertEqual(artifact["executor_provider"], "claude_code")
            self.assertTrue(artifact["real_executor_called"])
            self.assertEqual(len(store.list_record_ids("outcomes")), 1)


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


def _fake_claude_command(message: str) -> list[str]:
    return [
        sys.executable,
        "-c",
        (
            "import sys; "
            "prompt=sys.stdin.read(); "
            f"print({message!r} + ' prompt=' + str(len(prompt)))"
        ),
    ]


if __name__ == "__main__":
    unittest.main()
