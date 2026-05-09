from __future__ import annotations

import io
import json
import shlex
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timezone
from unittest.mock import patch

from spice.entry.cli import main as spice_cli_main
from spice.runtime import (
    LocalJsonStore,
    approve_approval,
    execute_hermes_approval,
    run_once,
    setup_workspace,
)
from spice.runtime.hermes_executor import execute_hermes_sdep_request
from spice.runtime.workspace import update_workspace_config


NOW = datetime(2026, 4, 29, 6, 0, tzinfo=timezone.utc)


class RuntimeHermesExecutorTests(unittest.TestCase):
    def test_hermes_sdep_endpoint_returns_valid_response(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, _approval_id, run_id = _approved_handoff(tmp_dir)
            request = store.load_run(run_id)["full_loop_preview"]["sdep_request"]

            response = execute_hermes_sdep_request(
                request,
                command=_fake_hermes_command("hermes fixture completed"),
            )

            self.assertEqual(response["message_type"], "execute.response")
            self.assertEqual(response["request_id"], request["request_id"])
            self.assertEqual(response["status"], "success")
            self.assertEqual(response["outcome"]["status"], "success")
            self.assertEqual(response["responder"]["id"], "hermes")
            self.assertEqual(response["metadata"]["executor_provider"], "hermes")
            self.assertEqual(
                response["traceability"]["approval_id"],
                request["traceability"]["approval_id"],
            )

    def test_hermes_chat_command_receives_prompt_as_query_argument(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, _approval_id, run_id = _approved_handoff(tmp_dir)
            request = store.load_run(run_id)["full_loop_preview"]["sdep_request"]

            with patch("spice.runtime.hermes_executor.subprocess.run") as run:
                run.return_value = subprocess.CompletedProcess(
                    args=["hermes", "chat", "-Q"],
                    returncode=0,
                    stdout="hermes fixture completed",
                    stderr="",
                )

                response = execute_hermes_sdep_request(
                    request,
                    command="hermes chat -Q",
                )

            called = run.call_args
            command = called.args[0]
            self.assertEqual(command[:3], ["hermes", "chat", "-Q"])
            self.assertIn("-q", command)
            self.assertIn("Fix the failing test.", command[command.index("-q") + 1])
            self.assertIsNone(called.kwargs["input"])
            self.assertEqual(response["outcome"]["status"], "success")

    def test_approved_approval_executes_through_hermes_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, approval_id, run_id = _approved_handoff(tmp_dir)

            result = execute_hermes_approval(
                approval_id,
                project_root=tmp_dir,
                command=_fake_hermes_command("hermes fixture completed"),
                now=NOW,
            )

            artifact = result.artifact
            run_payload = store.load_run(run_id)
            outcome = store.load_outcome(artifact["outcome_id"])
            state = store.load_state()
            general = state["world_state"]["domain_state"]["general_decision"]
            self.assertEqual(artifact["executor_provider"], "hermes")
            self.assertTrue(artifact["sdep_request_sent"])
            self.assertTrue(artifact["executor_called"])
            self.assertTrue(artifact["transport_executor_called"])
            self.assertTrue(artifact["real_executor_called"])
            self.assertEqual(artifact["protocol_status"], "success")
            self.assertEqual(artifact["task_status"], "success")
            self.assertEqual(outcome["executor_provider"], "hermes")
            self.assertEqual(run_payload["executor_provider"], "hermes")
            self.assertIn("hermes_execution", run_payload)
            self.assertEqual(len(general["outcomes"]), 1)

    def test_default_execute_dispatch_uses_hermes_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, approval_id, _run_id = _approved_handoff(tmp_dir)
            update_workspace_config(tmp_dir, "executor", "hermes")
            update_workspace_config(
                tmp_dir,
                "executor_command",
                shlex.join(_fake_hermes_command("hermes default dispatch")),
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
            self.assertEqual(artifact["executor_provider"], "hermes")
            self.assertEqual(len(store.list_record_ids("outcomes")), 1)

    def test_cli_execute_hermes_override_outputs_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            store, approval_id, _run_id = _approved_handoff(tmp_dir)
            stdout = io.StringIO()

            with redirect_stdout(stdout):
                exit_code = spice_cli_main(
                    [
                        "execute",
                        "hermes",
                        approval_id,
                        "--workspace",
                        tmp_dir,
                        "--command",
                        shlex.join(_fake_hermes_command("hermes explicit override")),
                        "--json",
                    ]
                )

            self.assertEqual(exit_code, 0)
            artifact = json.loads(stdout.getvalue())
            self.assertEqual(artifact["executor_provider"], "hermes")
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


def _fake_hermes_command(message: str) -> list[str]:
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
