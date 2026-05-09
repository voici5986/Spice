from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from spice.entry.cli import main as spice_cli_main
from spice.runtime import LocalJsonStore, perceive_once, setup_workspace


class RuntimePerceiveTests(unittest.TestCase):
    def test_perceive_once_polls_url_and_updates_general_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            source = Path(tmp_dir) / "status.txt"
            source.write_text("ci: failing\n", encoding="utf-8")

            result = perceive_once(project_root=tmp_dir, poll_url=source.as_uri())

            artifact = result.artifact
            self.assertEqual(artifact["path_type"], "runtime_perception_poll")
            self.assertEqual(artifact["provider"], "poll")
            self.assertEqual(artifact["observation_count"], 1)
            self.assertEqual(artifact["changed_count"], 1)
            self.assertFalse(artifact["decision_triggered"])
            self.assertFalse(artifact["executor_called"])
            self.assertFalse(artifact["sdep_request_sent"])
            self.assertTrue(artifact["state_updated"])
            self.assertTrue(result.perception_path.exists())
            self.assertTrue((Path(tmp_dir) / ".spice" / "perceptions" / "poll_state.json").exists())

            state = LocalJsonStore.from_project_root(tmp_dir).load_state()
            general = state["world_state"]["domain_state"]["general_decision"]
            self.assertEqual(len(general["signals"]), 1)
            self.assertEqual(general["signals"][0]["source"], "poll")
            json.dumps(artifact)

    def test_perceive_once_dedupes_unchanged_poll_without_duplicate_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            source = Path(tmp_dir) / "status.txt"
            source.write_text("same output\n", encoding="utf-8")

            first = perceive_once(project_root=tmp_dir, poll_url=source.as_uri())
            second = perceive_once(project_root=tmp_dir, poll_url=source.as_uri())

            self.assertEqual(first.artifact["observation_count"], 1)
            self.assertEqual(second.artifact["observation_count"], 0)
            self.assertEqual(second.artifact["deduped_count"], 1)
            self.assertFalse(second.artifact["state_updated"])
            state = LocalJsonStore.from_project_root(tmp_dir).load_state()
            general = state["world_state"]["domain_state"]["general_decision"]
            self.assertEqual(len(general["signals"]), 1)

    def test_perceive_once_decide_on_change_creates_decision_and_pending_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            source = Path(tmp_dir) / "status.txt"
            source.write_text("ci failed on main\n", encoding="utf-8")

            result = perceive_once(
                project_root=tmp_dir,
                poll_url=source.as_uri(),
                decide_on_change=True,
            )
            artifact = result.artifact
            store = LocalJsonStore.from_project_root(tmp_dir)

            self.assertEqual(artifact["trigger_mode"], "decision_on_change")
            self.assertTrue(artifact["decision_triggered"])
            self.assertTrue(artifact["run_id"])
            self.assertTrue(artifact["decision_id"])
            self.assertTrue(artifact["approval_id"])
            self.assertIsInstance(artifact["triggered_run"], dict)
            self.assertEqual(artifact["triggered_run"]["source"], "perception_trigger")
            self.assertEqual(artifact["triggered_run"]["input"]["kind"], "poll_signal")
            self.assertEqual(artifact["triggered_run"]["approval"]["status"], "pending")
            self.assertIn("decision_triggered: true", artifact["rendered_text"])
            self.assertFalse(artifact["executor_called"])
            self.assertFalse(artifact["sdep_request_sent"])
            self.assertTrue(store.record_path("run", artifact["run_id"]).exists())
            self.assertTrue(store.record_path("decision", artifact["decision_id"]).exists())
            self.assertTrue(store.record_path("approval", artifact["approval_id"]).exists())
            state = store.load_state()
            general = state["world_state"]["domain_state"]["general_decision"]
            self.assertEqual(len(general["signals"]), 1)
            self.assertEqual(len(general["intents"]), 1)
            self.assertEqual(len(general["decision_checkpoints"]), 1)

    def test_perceive_once_decision_on_change_does_not_trigger_when_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            source = Path(tmp_dir) / "status.txt"
            source.write_text("same output\n", encoding="utf-8")

            first = perceive_once(
                project_root=tmp_dir,
                poll_url=source.as_uri(),
                decide_on_change=True,
            )
            second = perceive_once(
                project_root=tmp_dir,
                poll_url=source.as_uri(),
                decide_on_change=True,
            )

            self.assertTrue(first.artifact["decision_triggered"])
            self.assertFalse(second.artifact["decision_triggered"])
            self.assertIsNone(second.artifact["triggered_run"])
            self.assertIsNone(second.artifact["approval_id"])
            self.assertEqual(second.artifact["observation_count"], 0)

    def test_perceive_once_configured_decision_on_change_triggers_without_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            config_path = Path(tmp_dir) / ".spice" / "config.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            config["perception_trigger_mode"] = "decision_on_change"
            config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            source = Path(tmp_dir) / "status.txt"
            source.write_text("new external signal\n", encoding="utf-8")

            result = perceive_once(project_root=tmp_dir, poll_url=source.as_uri())

            self.assertEqual(result.artifact["trigger_mode"], "decision_on_change")
            self.assertTrue(result.artifact["decision_triggered"])
            self.assertTrue(result.artifact["approval_id"])

    def test_perceive_once_command_poll_requires_allow_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            with self.assertRaisesRegex(PermissionError, "Command poll is disabled"):
                perceive_once(
                    project_root=tmp_dir,
                    poll_command=f"{sys.executable} -c \"print('ok')\"",
                )

    def test_perceive_once_command_poll_with_allow_updates_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)

            result = perceive_once(
                project_root=tmp_dir,
                poll_command=f"{sys.executable} -c \"print('tests changed')\"",
                allow_command_poll=True,
            )

            self.assertEqual(result.artifact["observation_count"], 1)
            observation = result.artifact["observations"][0]
            self.assertEqual(observation["source"]["channel"], "command")
            self.assertEqual(observation["attributes"]["kind"], "command_changed")

    def test_perceive_cli_json_outputs_parseable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "status.txt"
            source.write_text("deploy pending\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                setup_code = spice_cli_main(["setup", "--workspace", tmp_dir])
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                perceive_code = spice_cli_main(
                    [
                        "perceive",
                        "--workspace",
                        tmp_dir,
                        "--poll-url",
                        source.as_uri(),
                        "--json",
                    ]
                )

            self.assertEqual(setup_code, 0)
            self.assertEqual(perceive_code, 0)
            artifact = json.loads(stdout.getvalue())
            self.assertEqual(artifact["path_type"], "runtime_perception_poll")
            self.assertEqual(artifact["observation_count"], 1)

    def test_perceive_cli_decide_on_change_outputs_decision_refs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            source = Path(tmp_dir) / "status.txt"
            source.write_text("review needed\n", encoding="utf-8")
            with redirect_stdout(io.StringIO()):
                setup_code = spice_cli_main(["setup", "--workspace", tmp_dir])
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                perceive_code = spice_cli_main(
                    [
                        "perceive",
                        "--workspace",
                        tmp_dir,
                        "--poll-url",
                        source.as_uri(),
                        "--decide-on-change",
                        "--json",
                    ]
                )

            self.assertEqual(setup_code, 0)
            self.assertEqual(perceive_code, 0)
            artifact = json.loads(stdout.getvalue())
            self.assertTrue(artifact["decision_triggered"])
            self.assertTrue(artifact["approval_id"])
            self.assertFalse(artifact["executor_called"])
            self.assertFalse(artifact["sdep_request_sent"])

    def test_perceive_cli_command_poll_without_allow_has_friendly_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with redirect_stdout(io.StringIO()):
                setup_code = spice_cli_main(["setup", "--workspace", tmp_dir])
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                perceive_code = spice_cli_main(
                    [
                        "perceive",
                        "--workspace",
                        tmp_dir,
                        "--poll-command",
                        f"{sys.executable} -c \"print('ok')\"",
                    ]
                )

            self.assertEqual(setup_code, 0)
            self.assertEqual(perceive_code, 1)
            self.assertIn("perceive failed", stderr.getvalue())
            self.assertIn("--allow-command-poll", stderr.getvalue())

    def test_perceive_once_open_chronicle_updates_state(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            with _patched_open_chronicle_urlopen(
                "Current desktop context: VS Code - spice/runtime/perceive.py",
                "Recent activity: edited perception provider",
            ):
                result = perceive_once(
                    project_root=tmp_dir,
                    provider="open_chronicle",
                    openchronicle_mcp_url="http://127.0.0.1:8742/mcp",
                )

            artifact = result.artifact
            self.assertEqual(artifact["path_type"], "runtime_perception_open_chronicle")
            self.assertEqual(artifact["provider"], "open_chronicle")
            self.assertEqual(artifact["observation_count"], 2)
            self.assertEqual(artifact["changed_count"], 2)
            self.assertFalse(artifact["decision_triggered"])
            self.assertTrue(
                (Path(tmp_dir) / ".spice" / "perceptions" / "open_chronicle_state.json").exists()
            )
            state = LocalJsonStore.from_project_root(tmp_dir).load_state()
            general = state["world_state"]["domain_state"]["general_decision"]
            self.assertEqual(len(general["signals"]), 2)
            self.assertEqual(general["signals"][0]["source"], "open_chronicle")

    def test_perceive_once_open_chronicle_decide_on_change_creates_pending_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            with _patched_open_chronicle_urlopen(
                "Current desktop context: failing tests in terminal",
                "Recent activity: CI failed on main",
            ):
                result = perceive_once(
                    project_root=tmp_dir,
                    provider="open_chronicle",
                    openchronicle_mcp_url="http://127.0.0.1:8742/mcp",
                    decide_on_change=True,
                )

            artifact = result.artifact
            self.assertTrue(artifact["decision_triggered"])
            self.assertEqual(artifact["triggered_run"]["input"]["kind"], "open_chronicle_signal")
            self.assertEqual(artifact["triggered_run"]["input"]["source"], "open_chronicle")
            self.assertEqual(artifact["triggered_run"]["approval"]["status"], "pending")
            self.assertTrue(artifact["approval_id"])
            self.assertFalse(artifact["executor_called"])
            self.assertFalse(artifact["sdep_request_sent"])

    def test_perceive_cli_open_chronicle_json_outputs_parseable_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            with redirect_stdout(io.StringIO()):
                setup_code = spice_cli_main(["setup", "--workspace", tmp_dir])
            stdout = io.StringIO()
            with _patched_open_chronicle_urlopen("context", "activity"):
                with redirect_stdout(stdout):
                    perceive_code = spice_cli_main(
                        [
                            "perceive",
                            "--workspace",
                            tmp_dir,
                            "--provider",
                            "open_chronicle",
                            "--json",
                        ]
                    )

            self.assertEqual(setup_code, 0)
            self.assertEqual(perceive_code, 0)
            artifact = json.loads(stdout.getvalue())
            self.assertEqual(artifact["path_type"], "runtime_perception_open_chronicle")
            self.assertEqual(artifact["provider"], "open_chronicle")
            self.assertEqual(artifact["observation_count"], 2)


def _patched_open_chronicle_urlopen(current_context: str, recent_activity: str):
    responses = [
        _FakeMCPResponse(current_context),
        _FakeMCPResponse(recent_activity),
    ]
    return patch(
        "spice.perception.providers.open_chronicle.urllib.request.urlopen",
        side_effect=responses,
    )


class _FakeMCPResponse:
    def __init__(self, text: str) -> None:
        self.text = text

    def __enter__(self) -> "_FakeMCPResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self, _limit: int) -> bytes:
        return json.dumps(
            {
                "jsonrpc": "2.0",
                "id": "spice.open_chronicle.test",
                "result": {"content": [{"type": "text", "text": self.text}]},
            }
        ).encode("utf-8")


if __name__ == "__main__":
    unittest.main()
