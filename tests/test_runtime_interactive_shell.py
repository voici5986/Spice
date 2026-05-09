from __future__ import annotations

import io
import tempfile
import unittest

from spice.runtime import (
    LocalJsonStore,
    run_interactive_shell,
    setup_workspace,
)


class RuntimeInteractiveShellTests(unittest.TestCase):
    def test_shell_runs_one_default_intent_in_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()

            result = run_interactive_shell(
                project_root=tmp_dir,
                input_stream=io.StringIO("Review this repo and pick the safest next action.\n/exit\n"),
                output_stream=output,
                use_bars=False,
            )

            store = LocalJsonStore.from_project_root(tmp_dir)
            session = store.load_session("session.default")
            text = output.getvalue()
            self.assertEqual(result.status, "closed")
            self.assertEqual(result.turns, 1)
            self.assertEqual(len(result.run_ids), 1)
            self.assertEqual(session["last_run_id"], result.run_ids[0])
            self.assertIn("Spice Agent", text)
            self.assertIn("SPICE DECISION LOOP", text)
            self.assertIn("Artifacts:", text)

    def test_shell_act_lists_approval_and_preserves_pending_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()

            result = run_interactive_shell(
                project_root=tmp_dir,
                input_stream=io.StringIO("/act Fix the failing test.\n/approvals\n/exit\n"),
                output_stream=output,
                use_bars=False,
            )

            store = LocalJsonStore.from_project_root(tmp_dir)
            approvals = store.list_record_ids("approvals")
            session = store.load_session("session.default")
            text = output.getvalue()
            self.assertEqual(result.turns, 1)
            self.assertEqual(len(approvals), 1)
            self.assertEqual(session["pending_approval_ids"], approvals)
            self.assertIn("SPICE APPROVALS", text)
            self.assertIn("status=pending", text)

    def test_shell_can_approve_and_dry_run_existing_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            run_interactive_shell(
                project_root=tmp_dir,
                input_stream=io.StringIO("/act Fix the failing test.\n/exit\n"),
                output_stream=io.StringIO(),
                use_bars=False,
            )
            store = LocalJsonStore.from_project_root(tmp_dir)
            approval_id = store.list_record_ids("approvals")[0]
            output = io.StringIO()

            result = run_interactive_shell(
                project_root=tmp_dir,
                input_stream=io.StringIO(f"/approve {approval_id}\n/dry-run {approval_id}\n/exit\n"),
                output_stream=output,
                use_bars=False,
            )

            outcomes = store.list_record_ids("outcomes")
            session = store.load_session("session.default")
            text = output.getvalue()
            self.assertEqual(result.approved_ids, [approval_id])
            self.assertEqual(len(outcomes), 1)
            self.assertEqual(result.dry_run_outcome_ids, outcomes)
            self.assertEqual(session["pending_approval_ids"], [])
            self.assertEqual(session["metadata"]["last_outcome_id"], outcomes[0])
            self.assertIn("SPICE APPROVAL UPDATED", text)
            self.assertIn("SPICE DRY-RUN EXECUTION", text)
            self.assertIn("real_executor_called: false", text)

    def test_shell_advise_command_stops_at_decision_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()

            result = run_interactive_shell(
                project_root=tmp_dir,
                input_stream=io.StringIO("/advise What should I do next?\n/exit\n"),
                output_stream=output,
                use_bars=False,
            )

            text = output.getvalue()
            self.assertEqual(result.turns, 1)
            self.assertIn("SPICE RUN ONCE", text)
            self.assertNotIn("SPICE DECISION LOOP", text)

    def test_shell_refine_updates_latest_decision_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()

            result = run_interactive_shell(
                project_root=tmp_dir,
                input_stream=io.StringIO(
                    "Review the repo and pick the safest next action.\n"
                    "/refine Consider rollback first.\n"
                    "/exit\n"
                ),
                output_stream=output,
                use_bars=False,
            )

            store = LocalJsonStore.from_project_root(tmp_dir)
            session = store.load_session("session.default")
            text = output.getvalue()
            self.assertEqual(result.turns, 2)
            self.assertEqual(len(result.run_ids), 2)
            self.assertEqual(session["last_run_id"], result.run_ids[-1])
            self.assertIn("SPICE REFINE", text)
            self.assertIn("UPDATED DECISION CARD", text)

    def test_shell_help_and_unknown_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()

            result = run_interactive_shell(
                project_root=tmp_dir,
                input_stream=io.StringIO("/help\n/unknown\n/exit\n"),
                output_stream=output,
                use_bars=False,
            )

            text = output.getvalue()
            self.assertEqual(result.turns, 0)
            self.assertIn("/doctor", text)
            self.assertIn("/state", text)
            self.assertIn("/execute <id>", text)
            self.assertIn("/perceive", text)
            self.assertIn("/dry-run <id>", text)
            self.assertIn("unknown command: /unknown", text)

    def test_shell_supports_runtime_status_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()

            result = run_interactive_shell(
                project_root=tmp_dir,
                input_stream=io.StringIO("/doctor\n/state\n/timeline\n/stats\n/exit\n"),
                output_stream=output,
                use_bars=False,
            )

            text = output.getvalue()
            self.assertEqual(result.turns, 0)
            self.assertIn("Spice Doctor - workspace check", text)
            self.assertIn("WORLD STATE", text)
            self.assertIn("SPICE SESSION TIMELINE", text)
            self.assertIn("SPICE SESSION STATS", text)

    def test_shell_execute_uses_configured_executor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            run_interactive_shell(
                project_root=tmp_dir,
                input_stream=io.StringIO("/act Fix the failing test.\n/exit\n"),
                output_stream=io.StringIO(),
                use_bars=False,
            )
            store = LocalJsonStore.from_project_root(tmp_dir)
            approval_id = store.list_record_ids("approvals")[0]
            output = io.StringIO()

            result = run_interactive_shell(
                project_root=tmp_dir,
                input_stream=io.StringIO(f"/approve {approval_id}\n/execute {approval_id}\n/exit\n"),
                output_stream=output,
                use_bars=False,
            )

            text = output.getvalue()
            self.assertEqual(result.approved_ids, [approval_id])
            self.assertEqual(len(result.dry_run_outcome_ids), 1)
            self.assertIn("SPICE DRY-RUN EXECUTION", text)


if __name__ == "__main__":
    unittest.main()
