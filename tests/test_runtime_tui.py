from __future__ import annotations

import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from spice.runtime import LocalJsonStore, setup_workspace, update_workspace_config
from spice.runtime.approval_flow import load_approval
from spice.runtime.continuation_resolver import ContinuationResolution
from spice.runtime.tui.shell import SpiceTUIShell, run_tui_shell
from spice.runtime.tui.surfaces.banner import render_banner
from spice.runtime.tui.theme import COMMANDS


class RuntimeTUITests(unittest.TestCase):
    def test_run_tui_shell_plain_uses_plain_shell(self) -> None:
        with patch("spice.runtime.tui.shell.run_interactive_shell") as plain:
            plain.return_value = object()

            result = run_tui_shell(project_root=".", plain=True)

            self.assertIs(result, plain.return_value)
            plain.assert_called_once()

    def test_run_tui_shell_falls_back_when_prompt_toolkit_missing(self) -> None:
        with patch("spice.runtime.tui.shell._prompt_toolkit_available", return_value=False):
            with patch("spice.runtime.tui.shell.run_interactive_shell") as plain:
                plain.return_value = object()

                result = run_tui_shell(project_root=".")

                self.assertIs(result, plain.return_value)
                plain.assert_called_once()

    def test_banner_plain_fallback_contains_runtime_metadata(self) -> None:
        def blocked_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "rich" or name.startswith("rich."):
                raise ImportError("blocked")
            return __import__(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=blocked_import):
            banner = render_banner(
                {"executor": "dry_run", "llm_provider": "deterministic", "perception_provider": "manual"},
                {"session_id": "session.default"},
            )

        self.assertIn("Spice Decision Runtime", str(banner))
        self.assertIn("executor: dry_run", str(banner))
        self.assertIn("Available Executors", str(banner))
        self.assertIn("Available Skills", str(banner))

    def test_banner_dashboard_lists_runtime_capabilities(self) -> None:
        from rich.console import Console

        renderable = render_banner(
            {
                "executor": "dry_run",
                "llm_provider": "deterministic",
                "perception_provider": "manual",
            },
            {"session_id": "session.default"},
            dashboard={
                "mode": "decision + dry-run",
                "pending_approvals": 1,
                "decision_count": 3,
                "state_counts": {"work_items": 2, "outcomes": 1},
                "executors": [
                    {"name": "dry_run", "status": "ready"},
                    {"name": "sdep_subprocess", "status": "needs executor_command"},
                ],
                "skills": [
                    {"name": "item.triage", "status": "ready"},
                    {"name": "intent.execute", "status": "ready"},
                ],
                "perception": [
                    {"name": "manual", "status": "ready"},
                    {"name": "poll", "status": "needs poll source"},
                ],
            },
        )
        console = Console(file=io.StringIO(), record=True, force_terminal=False, width=120)
        console.print(renderable)
        text = console.export_text()

        self.assertIn("RUNTIME READINESS", text)
        self.assertIn("Available Executors", text)
        self.assertIn("dry_run", text)
        self.assertIn("Available Skills", text)
        self.assertIn("item.triage", text)
        self.assertIn("Perception", text)
        self.assertIn("pending", text)
        self.assertIn("/pending", text)

    def test_banner_uses_compact_header_when_terminal_is_narrow(self) -> None:
        from rich.console import Console

        renderable = render_banner(
            {
                "executor": "dry_run",
                "llm_provider": "deterministic",
                "perception_provider": "manual",
            },
            {"session_id": "session.default"},
            width=38,
        )
        console = Console(file=io.StringIO(), record=True, force_terminal=False, width=38)
        console.print(renderable)
        text = console.export_text()

        self.assertIn("compact", text)
        self.assertIn("banner", text)
        self.assertNotIn("██████", text)

    def test_tui_shell_refresh_redraws_banner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)
            shell.store = LocalJsonStore.from_project_root(tmp_dir)

            should_exit = shell.handle_line("/refresh")

            self.assertFalse(should_exit)
            self.assertIn("Spice Decision Runtime", output.getvalue())

    def test_tui_shell_handles_intent_with_rich_decision_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            should_exit = shell.handle_line("Review the repo and pick the safest next action.")

            self.assertFalse(should_exit)
            text = output.getvalue()
            self.assertIn("SPICE DECISION CARD", text)
            self.assertIn("Artifacts:", text)
            self.assertEqual(shell.result.turns, 1)

    def test_tui_shell_refine_updates_latest_decision_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line("Review the repo and pick the safest next action.")
            self.assertFalse(shell.handle_line("/refine Consider rollback first."))

            text = output.getvalue()
            self.assertIn("SPICE DECISION CARD", text)
            self.assertIn("Refine artifacts:", text)
            self.assertEqual(shell.result.turns, 2)

    def test_tui_shell_enters_decision_feedback_after_action_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            self.assertFalse(shell.handle_line("/act Fix the failing test."))

            text = output.getvalue()
            self.assertIn("Decision prompt shortcuts", text)
            self.assertIn("y / yes", text)
            self.assertIn("approve and execute with the configured executor", text)
            self.assertIn("decision> y", text)
            self.assertIn("decision> reject too risky right now", text)
            self.assertIn("decision> refine execute directly; do not split", text)
            self.assertIsNotNone(shell.pending_decision)
            self.assertEqual(shell._prompt_text(), "decision> ")

    def test_tui_shell_approve_only_uses_latest_pending_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line("/act Fix the failing test.")
            approval_id = str(shell.pending_decision["approval_id"])
            self.assertFalse(shell.handle_line("a"))

            text = output.getvalue()
            self.assertIn("APPROVAL APPROVED", text)
            self.assertIn(approval_id, shell.result.approved_ids)
            self.assertIsNone(shell.pending_decision)
            self.assertEqual(shell._prompt_text(), "spice> ")

    def test_tui_shell_decision_action_picker_can_approve_latest_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            with patch.object(shell, "_decision_action_picker_available", return_value=True):
                with patch.object(shell, "_prompt_decision_action", return_value="approve") as picker:
                    self.assertFalse(shell.handle_line("/act Fix the failing test."))

            text = output.getvalue()
            self.assertIn("APPROVAL APPROVED", text)
            self.assertIsNone(shell.pending_decision)
            picker.assert_called_once()

    def test_tui_shell_decision_action_picker_is_available_for_tty_output(self) -> None:
        class TTYOutput(io.StringIO):
            def isatty(self) -> bool:
                return True

        with tempfile.TemporaryDirectory() as tmp_dir:
            output = TTYOutput()
            shell = self._plain_output_shell(tmp_dir, output)
            shell.console = object()

            with patch("spice.runtime.tui.shell._prompt_toolkit_available", return_value=True):
                self.assertTrue(shell._decision_action_picker_available())

    def test_tui_shell_yes_approves_and_executes_latest_pending_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line("/act Fix the failing test.")
            approval_id = str(shell.pending_decision["approval_id"])
            self.assertFalse(shell.handle_line("y"))

            text = output.getvalue()
            self.assertIn("APPROVAL APPROVED", text)
            self.assertIn("DISPATCHING TO EXECUTOR", text)
            self.assertIn("EXECUTION COMPLETE", text)
            self.assertIn(approval_id, shell.result.approved_ids)
            self.assertGreaterEqual(len(shell.result.dry_run_outcome_ids), 1)
            self.assertIsNone(shell.pending_decision)

    def test_tui_shell_rejects_latest_pending_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line("/act Fix the failing test.")
            approval_id = str(shell.pending_decision["approval_id"])
            self.assertFalse(shell.handle_line("n too risky right now"))

            text = output.getvalue()
            self.assertIn("APPROVAL REJECTED", text)
            self.assertIn(approval_id, shell.result.rejected_ids)
            self.assertIsNone(shell.pending_decision)

    def test_tui_shell_feedback_refines_latest_pending_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line("/act Fix the failing test.")
            self.assertFalse(shell.handle_line("consider rollback first"))

            text = output.getvalue()
            self.assertIn("Refine artifacts:", text)
            self.assertEqual(shell.result.turns, 2)
            self.assertIsNotNone(shell.pending_decision)

    def test_tui_shell_doctor_and_state_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            self.assertFalse(shell.handle_line("/doctor"))
            self.assertFalse(shell.handle_line("/state"))

            text = output.getvalue()
            self.assertIn("SPICE DOCTOR", text)
            self.assertIn("WORLD STATE", text)

    def test_tui_shell_context_command_renders_compiled_decision_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            self.assertFalse(shell.handle_line("/context"))

            text = output.getvalue()
            self.assertIn("COMPILED DECISION CONTEXT", text)
            self.assertIn("context_type", text)
            self.assertIn("decision", text)
            self.assertIn("workspace", text)

    def test_tui_shell_context_json_outputs_exact_context_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            self.assertFalse(shell.handle_line("/context --json"))

            payload = json.loads(output.getvalue())
            self.assertEqual(payload["context_type"], "decision")
            self.assertEqual(payload["domain"], "general")
            self.assertEqual(payload["workspace_context"]["memory_provider"], "file")
            self.assertEqual(payload["workspace_context"]["context_compiler"], "deterministic")
            self.assertIn("retrieved_memory", payload)

    def test_tui_shell_context_json_includes_active_decision_frame_after_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            setup_output = io.StringIO()
            setup_shell = self._plain_output_shell(tmp_dir, setup_output)
            setup_shell.handle_line("Review the repo and pick the safest next action.")

            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)
            self.assertFalse(shell.handle_line("/context --json"))

            payload = json.loads(output.getvalue())
            frame = payload["active_decision_frame"]
            self.assertTrue(frame["decision_id"])
            self.assertEqual(
                payload["current_intent"]["text"],
                "Review the repo and pick the safest next action.",
            )
            self.assertTrue(frame["candidates"])

    def test_tui_shell_perceive_command_updates_state_and_renders_panel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            source = Path(tmp_dir) / "status.txt"
            source.write_text("ci failed\n", encoding="utf-8")
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            self.assertFalse(shell.handle_line(f"/perceive --poll-url {source.as_uri()}"))

            text = output.getvalue()
            self.assertIn("PERCEPTION", text)
            self.assertIn("provider:", text)
            self.assertIn("poll", text)
            self.assertIn("Perception artifacts:", text)
            self.assertEqual(shell.result.turns, 1)

    def test_tui_shell_perceive_decide_on_change_records_triggered_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            source = Path(tmp_dir) / "status.txt"
            source.write_text("deployment needs review\n", encoding="utf-8")
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            self.assertFalse(shell.handle_line(f"/perceive --poll-url {source.as_uri()} --decide-on-change"))

            text = output.getvalue()
            self.assertIn("decision_triggered:", text)
            self.assertIn("approval_id:", text)
            self.assertEqual(len(shell.result.run_ids), 1)
            self.assertEqual(shell.result.turns, 1)

    def test_tui_shell_perceive_open_chronicle_uses_provider_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)
            with patch(
                "spice.runtime.tui.shell.perceive_once",
                return_value=SimpleNamespace(
                    artifact={
                        "provider": "open_chronicle",
                        "observation_count": 2,
                        "changed_count": 2,
                        "deduped_count": 0,
                        "decision_triggered": False,
                        "executor_called": False,
                        "sdep_request_sent": False,
                        "observations": [],
                    },
                    perception_path=Path(tmp_dir) / ".spice" / "perceptions" / "p.json",
                    state_path=Path(tmp_dir) / ".spice" / "state" / "state.json",
                ),
            ) as perceive:
                self.assertFalse(
                    shell.handle_line(
                        "/perceive --provider open_chronicle "
                        "--openchronicle-mcp-url http://127.0.0.1:8742/mcp "
                        "--openchronicle-since-minutes 10 "
                        "--openchronicle-context-limit 3"
                    )
                )

            kwargs = perceive.call_args.kwargs
            self.assertEqual(kwargs["provider"], "open_chronicle")
            self.assertEqual(kwargs["openchronicle_mcp_url"], "http://127.0.0.1:8742/mcp")
            self.assertEqual(kwargs["openchronicle_since_minutes"], 10)
            self.assertEqual(kwargs["openchronicle_context_limit"], 3)
            self.assertIn("PERCEPTION", output.getvalue())

    def test_tui_shell_approvals_and_session_commands(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line("/act Fix the failing test.")
            self.assertFalse(shell.handle_line("/approvals"))
            self.assertFalse(shell.handle_line("/session"))
            self.assertFalse(shell.handle_line("/timeline"))
            self.assertFalse(shell.handle_line("/stats"))

            text = output.getvalue()
            self.assertIn("APPROVALS", text)
            self.assertIn("SESSION:", text)
            self.assertIn("TIMELINE:", text)
            self.assertIn("SESSION STATS", text)

    def test_tui_shell_pending_without_pending_approvals_reports_none(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            self.assertFalse(shell.handle_line("/pending"))

            self.assertIn("No pending approvals.", output.getvalue())

    def test_tui_shell_pending_single_approval_opens_action_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line("/act Fix the failing test.")
            approval_id = str(shell.pending_decision["approval_id"])
            shell.pending_decision = None

            with patch.object(shell, "_decision_action_picker_available", return_value=True):
                with patch.object(shell, "_prompt_decision_action", return_value="approve") as picker:
                    self.assertFalse(shell.handle_line("/pending"))

            text = output.getvalue()
            self.assertIn(f"Pending approval: {approval_id}", text)
            self.assertIn("APPROVAL APPROVED", text)
            picker.assert_called_once()

    def test_tui_shell_pending_approve_execute_runs_configured_executor(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line("/act Fix the failing test.")
            approval_id = str(shell.pending_decision["approval_id"])
            shell.pending_decision = None

            with patch.object(shell, "_decision_action_picker_available", return_value=True):
                with patch.object(shell, "_prompt_decision_action", return_value="approve_execute"):
                    self.assertFalse(shell.handle_line("/pending"))

            text = output.getvalue()
            self.assertIn("APPROVAL APPROVED", text)
            self.assertIn("EXECUTION COMPLETE", text)
            self.assertIn(approval_id, shell.result.approved_ids)
            self.assertGreaterEqual(len(shell.result.dry_run_outcome_ids), 1)

    def test_tui_shell_approval_command_opens_specific_approval_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line("/act Fix the failing test.")
            approval_id = str(shell.pending_decision["approval_id"])
            shell.pending_decision = None

            with patch.object(shell, "_decision_action_picker_available", return_value=True):
                with patch.object(shell, "_prompt_decision_action", side_effect=["details", "approve"]) as picker:
                    self.assertFalse(shell.handle_line(f"/approval {approval_id}"))

            text = output.getvalue()
            self.assertIn("SPICE APPROVAL", text)
            self.assertIn("APPROVAL APPROVED", text)
            self.assertEqual(picker.call_count, 2)

    def test_tui_shell_approval_command_rejects_non_pending_approval_flow(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line("/act Fix the failing test.")
            approval_id = str(shell.pending_decision["approval_id"])
            shell.handle_line("a")

            self.assertFalse(shell.handle_line(f"/approval {approval_id}"))

            text = output.getvalue()
            self.assertIn("is not pending", text)
            self.assertIn("Current status: approved", text)

    def test_tui_shell_pending_multiple_approvals_selects_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line("/act Fix the failing test.")
            first_id = str(shell.pending_decision["approval_id"])
            shell.handle_line("/act Review the pending PR.")
            second_id = str(shell.pending_decision["approval_id"])
            shell.pending_decision = None

            with patch.object(shell, "_choose_pending_approval", return_value=second_id) as choose:
                with patch.object(shell, "_approval_action_menu") as menu:
                    self.assertFalse(shell.handle_line("/pending"))

            self.assertNotEqual(first_id, second_id)
            choose.assert_called_once()
            menu.assert_called_once_with(second_id)

    def test_tui_shell_continuation_selects_visible_decision_card_option(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line(
                "Compare these 3 next steps for Spice: add LLM retry, polish Decision Card, "
                "add JSON output. Which should we do first?"
            )
            before = shell._store().load_state()
            before_frame = before["world_state"]["domain_state"]["general_decision"]["metadata"][
                "active_decision_frame"
            ]
            visible_b = before_frame["candidates"][1]

            self.assertFalse(shell.handle_line("B"))

            after = shell._store().load_state()
            after_frame = after["world_state"]["domain_state"]["general_decision"]["metadata"][
                "active_decision_frame"
            ]
            self.assertEqual(after_frame["selected_candidate_id"], visible_b["candidate_id"])
            self.assertEqual(after_frame["selected"]["label"], "B")
            self.assertIn("Selected B", output.getvalue())

    def test_tui_shell_continuation_execute_selected_opens_act_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line(
                "Compare these 3 next steps for Spice: add LLM retry, polish Decision Card, "
                "add JSON output. Which should we do first?"
            )
            shell.handle_line("choose B")

            with patch.object(shell, "_run_intent") as run_intent:
                self.assertFalse(shell.handle_line("execute selected"))

            run_intent.assert_called_once()
            self.assertEqual(run_intent.call_args.kwargs["mode"], "act")
            self.assertIn("polish Decision Card", run_intent.call_args.args[0])

    def test_tui_shell_uses_llm_continuation_fallback_for_natural_followup(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line(
                "Compare these 3 next steps for Spice: add LLM retry, polish Decision Card, "
                "add JSON output. Which should we do first?"
            )

            with patch(
                "spice.runtime.tui.shell.resolve_continuation_from_runtime_config",
                return_value=ContinuationResolution(
                    True,
                    action="execute_selected",
                    candidate_id="candidate.test",
                    text="那就去干吧",
                ),
            ) as resolver:
                with patch.object(shell, "_run_intent") as run_intent:
                    self.assertFalse(shell.handle_line("那就去干吧"))

            resolver.assert_called_once()
            run_intent.assert_called_once()
            self.assertEqual(run_intent.call_args.kwargs["mode"], "act")

    def test_tui_shell_english_natural_execution_moves_advisory_to_approval(self) -> None:
        self._assert_natural_execution_followup_creates_approval("implement this")

    def test_tui_shell_chinese_natural_execution_moves_advisory_to_approval(self) -> None:
        self._assert_natural_execution_followup_creates_approval("那就开始做吧")

    def test_tui_shell_continuation_refines_active_decision_card(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line("Review the repo and pick the safest next action.")

            with patch.object(shell, "_refine_decision") as refine:
                self.assertFalse(shell.handle_line("refine that to lower risk"))

            refine.assert_called_once_with("to lower risk")

    def test_tui_shell_pending_decision_accepts_execute_selected_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line("/act Fix the failing test.")
            approval_id = str(shell.pending_decision["approval_id"])
            self.assertFalse(shell.handle_line("execute selected"))

            text = output.getvalue()
            self.assertIn("APPROVAL APPROVED", text)
            self.assertIn("EXECUTION COMPLETE", text)
            self.assertIn(approval_id, shell.result.approved_ids)
            self.assertIsNone(shell.pending_decision)

    def test_tui_execute_configured_renders_execution_panel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)
            execution = SimpleNamespace(
                artifact={
                    "executor_provider": "dry_run",
                    "approval_id": "approval.test",
                    "decision_id": "decision.test",
                    "trace_ref": "trace.test",
                    "candidate_id": "candidate.test",
                    "execution_id": "execution.test",
                    "request_id": "request.test",
                    "outcome_id": "outcome.test",
                    "executor_id": "spice.general_executor",
                    "skill_id": "runtime.intent.execute",
                    "context_pack_id": "context.test",
                    "sdep_request_sent": False,
                    "executor_called": False,
                    "real_executor_called": False,
                    "executed": False,
                    "protocol_status": "success",
                    "task_status": "success",
                    "state_updated": True,
                    "persisted": True,
                    "state_after_ref": ".spice/state/state.json#after:test",
                },
                rendered_text="SPICE DRY-RUN EXECUTION\nplain executor output",
            )

            with patch("spice.runtime.tui.shell.execute_dry_run_approval", return_value=execution):
                self.assertFalse(shell.handle_line("/execute approval.test"))

            text = output.getvalue()
            self.assertIn("DISPATCHING TO EXECUTOR", text)
            self.assertIn("EXECUTION COMPLETE", text)
            self.assertIn("finished successfully", text)
            self.assertIn("approval.test", text)
            self.assertNotIn("plain executor output", text)

    def test_tui_execute_configured_renders_execution_error_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            with patch("spice.runtime.tui.shell.execute_dry_run_approval", side_effect=TimeoutError("executor timed out")):
                self.assertFalse(shell.handle_line("/execute approval.test"))

            text = output.getvalue()
            self.assertIn("DISPATCHING TO EXECUTOR", text)
            self.assertIn("EXECUTION DID NOT COMPLETE", text)
            self.assertIn("timed out", text)
            self.assertIn("/execute approval.test", text)

    def test_tui_handles_pasted_multiline_commands_separately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            self.assertFalse(shell.handle_line("/approve approval.test\n/execute approval.test"))

            text = output.getvalue()
            self.assertIn("error:", text)
            self.assertNotIn("__execute", text)
            self.assertIn("/execute approval.test", text)

    def test_tui_rejects_extra_approval_id_arguments(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            self.assertFalse(shell.handle_line("/approve approval.one approval.two"))
            self.assertFalse(shell.handle_line("/execute approval.one approval.two"))

            text = output.getvalue()
            self.assertIn("/approve requires exactly one approval id", text)
            self.assertIn("/execute requires exactly one approval id", text)

    def test_codex_approve_execute_refuses_permission_escalation_keeps_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            update_workspace_config(tmp_dir, "executor", "codex")
            update_workspace_config(tmp_dir, "executor_permission_mode", "read_only")
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line(
                "/act Add a small smoke note file at .spice-smoke/codex_executor_smoke.txt."
            )
            approval_id = str(shell.pending_decision["approval_id"])

            with patch.object(shell, "_prompt_permission_escalation", return_value="no"):
                self.assertFalse(shell.handle_line("y"))

            approval = load_approval(shell._store(), approval_id)
            self.assertEqual(approval.status, "pending")
            self.assertIsNotNone(shell.pending_decision)
            self.assertIn("Approval remains pending", output.getvalue())

    def test_codex_approve_execute_escalates_permission_for_single_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            update_workspace_config(tmp_dir, "executor", "codex")
            update_workspace_config(tmp_dir, "executor_permission_mode", "read_only")
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line(
                "/act Add a small smoke note file at .spice-smoke/codex_executor_smoke.txt."
            )
            approval_id = str(shell.pending_decision["approval_id"])

            with patch.object(shell, "_prompt_permission_escalation", return_value="yes"):
                with patch.object(shell, "_execute_configured") as execute:
                    self.assertFalse(shell.handle_line("y"))

            approval = load_approval(shell._store(), approval_id)
            self.assertEqual(approval.status, "approved")
            execute.assert_called_once_with(approval_id, permission_mode="workspace_write")

    def test_claude_code_approve_execute_escalates_permission_for_single_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            update_workspace_config(tmp_dir, "executor", "claude_code")
            update_workspace_config(tmp_dir, "executor_permission_mode", "read_only")
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line(
                "/act Add a small smoke note file at .spice-smoke/claude_code_executor_smoke.txt."
            )
            approval_id = str(shell.pending_decision["approval_id"])

            with patch.object(shell, "_prompt_permission_escalation", return_value="yes"):
                with patch.object(shell, "_execute_configured") as execute:
                    self.assertFalse(shell.handle_line("y"))

            approval = load_approval(shell._store(), approval_id)
            self.assertEqual(approval.status, "approved")
            execute.assert_called_once_with(approval_id, permission_mode="workspace_write")

    def test_hermes_approve_execute_escalates_permission_for_single_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            update_workspace_config(tmp_dir, "executor", "hermes")
            update_workspace_config(tmp_dir, "executor_permission_mode", "read_only")
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line(
                "/act Add a small smoke note file at .spice-smoke/hermes_executor_smoke.txt."
            )
            approval_id = str(shell.pending_decision["approval_id"])

            with patch.object(shell, "_prompt_permission_escalation", return_value="yes"):
                with patch.object(shell, "_execute_configured") as execute:
                    self.assertFalse(shell.handle_line("y"))

            approval = load_approval(shell._store(), approval_id)
            self.assertEqual(approval.status, "approved")
            execute.assert_called_once_with(approval_id, permission_mode="workspace_write")

    def test_tui_command_completer_includes_perceive(self) -> None:
        self.assertIn("/perceive", COMMANDS)
        self.assertIn("/pending", COMMANDS)
        self.assertIn("/approval", COMMANDS)
        self.assertIn("/context", COMMANDS)

    def _assert_natural_execution_followup_creates_approval(self, followup: str) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            setup_workspace(project_root=tmp_dir)
            output = io.StringIO()
            shell = self._plain_output_shell(tmp_dir, output)

            shell.handle_line(
                "Compare these 3 next steps for Spice: add LLM retry, polish Decision Card, "
                "add JSON output. Which should we do first?"
            )
            before_state = shell._store().load_state()
            active_frame = before_state["world_state"]["domain_state"]["general_decision"]["metadata"][
                "active_decision_frame"
            ]
            self.assertFalse(active_frame.get("approval_id"))
            self.assertEqual(shell._store().list_record_ids("approvals"), [])
            self.assertIsNone(shell.pending_decision)

            with patch(
                "spice.runtime.tui.shell.resolve_continuation_from_runtime_config",
                return_value=ContinuationResolution(
                    True,
                    action="execute_selected",
                    candidate_id=str(active_frame["selected_candidate_id"]),
                    text=followup,
                    reason="LLM fallback classified this as an execution follow-up.",
                ),
            ) as resolver:
                self.assertFalse(shell.handle_line(followup))

            approval_ids = shell._store().list_record_ids("approvals")
            self.assertEqual(len(approval_ids), 1)
            self.assertIsNotNone(shell.pending_decision)
            self.assertEqual(shell.pending_decision["approval_id"], approval_ids[0])
            self.assertEqual(shell._prompt_text(), "decision> ")
            self.assertIn("approval_id:", output.getvalue())
            resolver.assert_called_once()

    def _plain_output_shell(self, tmp_dir: str, output: io.StringIO) -> SpiceTUIShell:
        with patch.object(SpiceTUIShell, "_build_prompt_session", return_value=object()):
            with patch.object(SpiceTUIShell, "_build_console", return_value=None):
                return SpiceTUIShell(
                    project_root=tmp_dir,
                    output_stream=output,
                    history_path=Path(tmp_dir) / "history",
                )


if __name__ == "__main__":
    unittest.main()
