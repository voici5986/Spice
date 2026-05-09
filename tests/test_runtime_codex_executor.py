from __future__ import annotations

import io
import json
import shlex
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path

from spice.entry.cli import main as spice_cli_main
from spice.runtime import (
    LocalJsonStore,
    approve_approval,
    execute_codex_approval,
    run_once,
    setup_workspace,
)
from spice.runtime.codex_executor import execute_codex_sdep_request
from spice.runtime.workspace import update_workspace_config


NOW = datetime(2026, 4, 29, 6, 0, tzinfo=timezone.utc)


class RuntimeCodexExecutorTests(unittest.TestCase):
    def test_codex_sdep_endpoint_returns_valid_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, _approval_id, run_id = _approved_handoff(tmp_dir)
            request = store.load_run(run_id)["full_loop_preview"]["sdep_request"]

            response = execute_codex_sdep_request(
                request,
                command=_fake_codex_command("codex fixture completed"),
            )

            self.assertEqual(response["message_type"], "execute.response")
            self.assertEqual(response["request_id"], request["request_id"])
            self.assertEqual(response["status"], "success")
            self.assertEqual(response["outcome"]["status"], "success")
            self.assertEqual(response["responder"]["id"], "codex")
            self.assertEqual(response["metadata"]["executor_provider"], "codex")
            self.assertEqual(
                response["traceability"]["approval_id"],
                request["traceability"]["approval_id"],
            )

    def test_codex_sdep_endpoint_propagates_reported_blocked_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, _approval_id, run_id = _approved_handoff(tmp_dir)
            request = store.load_run(run_id)["full_loop_preview"]["sdep_request"]

            response = execute_codex_sdep_request(
                request,
                command=_fake_codex_json_command(
                    {
                        "protocol_status": "completed",
                        "task_status": "blocked",
                        "output": "Write rejected by read-only sandbox.",
                        "state_delta": {"task_status": "blocked"},
                    }
                ),
            )

            self.assertEqual(response["status"], "blocked")
            self.assertEqual(response["outcome"]["status"], "blocked")
            self.assertEqual(response["outcome"]["outcome_type"], "error")
            self.assertEqual(
                response["outcome"]["output"]["state_delta"]["task_status"],
                "blocked",
            )

    def test_approved_approval_executes_through_codex_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, approval_id, run_id = _approved_handoff(tmp_dir)

            result = execute_codex_approval(
                approval_id,
                project_root=tmp_dir,
                command=_fake_codex_command("codex fixture completed"),
                now=NOW,
            )

            artifact = result.artifact
            run_payload = store.load_run(run_id)
            outcome = store.load_outcome(artifact["outcome_id"])
            state = store.load_state()
            general = state["world_state"]["domain_state"]["general_decision"]
            self.assertEqual(artifact["executor_provider"], "codex")
            self.assertTrue(artifact["sdep_request_sent"])
            self.assertTrue(artifact["executor_called"])
            self.assertTrue(artifact["transport_executor_called"])
            self.assertTrue(artifact["real_executor_called"])
            self.assertEqual(artifact["protocol_status"], "success")
            self.assertEqual(artifact["task_status"], "success")
            self.assertEqual(outcome["executor_provider"], "codex")
            self.assertEqual(run_payload["executor_provider"], "codex")
            self.assertIn("codex_execution", run_payload)
            self.assertEqual(len(general["outcomes"]), 1)

    def test_default_execute_dispatch_uses_codex_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, approval_id, _run_id = _approved_handoff(tmp_dir)
            update_workspace_config(tmp_dir, "executor", "codex")
            update_workspace_config(
                tmp_dir,
                "executor_command",
                shlex.join(_fake_codex_command("codex default dispatch")),
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
            self.assertEqual(artifact["executor_provider"], "codex")
            self.assertEqual(len(store.list_record_ids("outcomes")), 1)

    def test_cli_execute_codex_override_outputs_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, approval_id, _run_id = _approved_handoff(tmp_dir)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = spice_cli_main(
                    [
                        "execute",
                        "codex",
                        approval_id,
                        "--workspace",
                        tmp_dir,
                        "--command",
                        shlex.join(_fake_codex_command("codex explicit override")),
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            artifact = json.loads(stdout.getvalue())
            self.assertEqual(artifact["executor_provider"], "codex")
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


def _fake_codex_command(message: str) -> list[str]:
    return [
        sys.executable,
        "-c",
        (
            "import sys; "
            "prompt=sys.stdin.read(); "
            f"print({message!r} + ' prompt=' + str(len(prompt)))"
        ),
    ]


def _fake_codex_json_command(payload: dict[str, object]) -> list[str]:
    return [
        sys.executable,
        "-c",
        (
            "import json, sys; "
            "sys.stdin.read(); "
            f"print(json.dumps({payload!r}, sort_keys=True))"
        ),
    ]


if __name__ == "__main__":
    unittest.main()
