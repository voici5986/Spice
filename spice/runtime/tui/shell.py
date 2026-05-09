from __future__ import annotations

import json
import shutil
import shlex
import sys
from pathlib import Path
from typing import Any, TextIO

from spice.decision.general import load_general_state
from spice.decision.general.types import payload_value
from spice.protocols import WorldState
from spice.runtime.approval_flow import (
    approve_approval,
    list_approvals,
    load_approval,
    reject_approval,
)
from spice.runtime.claude_code_provider import execute_claude_code_approval
from spice.runtime.continuation_resolver import (
    ContinuationResolution,
    resolve_continuation_from_runtime_config,
    selected_candidate_execution_text,
    update_frame_selected_candidate,
)
from spice.runtime.codex_provider import execute_codex_approval
from spice.runtime.dry_run_executor import execute_dry_run_approval
from spice.runtime.doctor import run_doctor
from spice.runtime.execution_permission import approval_requires_permission_escalation
from spice.runtime.executor_runtime import (
    resolve_executor_runtime_from_config,
    resolve_executor_runtime_from_config_with_permission,
)
from spice.runtime.hermes_provider import execute_hermes_approval
from spice.runtime.interactive_shell import InteractiveShellResult, run_interactive_shell
from spice.runtime.perceive import perceive_once
from spice.runtime.run_once import RunOnceResult, run_once
from spice.runtime.refine import RefineResult, refine_decision
from spice.runtime.sdep_subprocess_executor import execute_sdep_subprocess_approval
from spice.runtime.session import (
    DEFAULT_SESSION_ID,
    build_session_timeline,
    load_or_create_session,
    session_stats,
)
from spice.runtime.store import LocalJsonStore
from spice.runtime.tui.surfaces.approval import (
    render_approval_details_panel,
    render_approval_resolution_panel,
    render_approvals_panel,
)
from spice.runtime.tui.surfaces.banner import render_banner
from spice.runtime.tui.surfaces.decisioncard import render_decision_card
from spice.runtime.tui.surfaces.doctor import render_doctor_panel
from spice.runtime.tui.surfaces.execution import render_execution_panel
from spice.runtime.tui.surfaces.execution import render_execution_dispatch_panel
from spice.runtime.tui.surfaces.execution import render_execution_error_panel
from spice.runtime.tui.surfaces.execution import render_execution_summary_panel
from spice.runtime.tui.surfaces.perception import render_perception_panel
from spice.runtime.tui.surfaces.progress import decision_progress, execution_progress
from spice.runtime.tui.surfaces.session import render_session_panel, render_stats_panel, render_timeline_panel
from spice.runtime.tui.surfaces.state import render_state_panel
from spice.runtime.tui.theme import COMMANDS, SpiceTheme
from spice.runtime.workspace import SpiceWorkspaceConfig, load_workspace_config, require_workspace
from spice.runtime.workspace import load_workspace_context_compiler


def run_tui_shell(
    *,
    project_root: str | Path = ".",
    session_id: str = DEFAULT_SESSION_ID,
    plain: bool = False,
    input_stream: TextIO | None = None,
    output_stream: TextIO | None = None,
    use_bars: bool = True,
    persist: bool = True,
    full_loop_preview: bool = True,
    run_intent_mode: str = "auto",
) -> InteractiveShellResult:
    if plain or not _prompt_toolkit_available():
        return run_interactive_shell(
            project_root=project_root,
            session_id=session_id,
            input_stream=input_stream,
            output_stream=output_stream,
            use_bars=use_bars,
            persist=persist,
            full_loop_preview=full_loop_preview,
            run_intent_mode=run_intent_mode,
        )
    shell = SpiceTUIShell(
        project_root=project_root,
        session_id=session_id,
        output_stream=output_stream,
        use_bars=use_bars,
        persist=persist,
        full_loop_preview=full_loop_preview,
        run_intent_mode=run_intent_mode,
    )
    return shell.run()


class SpiceTUIShell:
    def __init__(
        self,
        *,
        project_root: str | Path = ".",
        session_id: str = DEFAULT_SESSION_ID,
        output_stream: TextIO | None = None,
        history_path: Path | None = None,
        use_bars: bool = True,
        persist: bool = True,
        full_loop_preview: bool = True,
        run_intent_mode: str = "auto",
    ) -> None:
        self.project_root = Path(project_root)
        self.session_id = session_id
        self.output_stream = output_stream
        self.use_bars = use_bars
        self.persist = persist
        self.full_loop_preview = full_loop_preview
        self.run_intent_mode = run_intent_mode
        self.history_path = history_path or (Path.home() / ".spice" / ".history")
        self.console = self._build_console(output_stream)
        self.prompt_session = self._build_prompt_session()
        self.store: LocalJsonStore | None = None
        self.result = InteractiveShellResult(session_id=session_id)
        self.pending_decision: dict[str, str] | None = None

    def run(self) -> InteractiveShellResult:
        paths = require_workspace(self.project_root)
        self.store = LocalJsonStore(paths)
        config = load_workspace_config(self.project_root).to_payload()
        session = load_or_create_session(self.store, session_id=self.session_id)
        self.store.save_session(session.session_id, session.to_payload())
        self.result.session_id = session.session_id
        self._render_startup_banner(config=config, session_payload=session.to_payload())
        self.print(self.render_help(compact=True))
        while True:
            try:
                line = self.prompt_session.prompt([("class:prompt", self._prompt_text())])
            except (EOFError, KeyboardInterrupt):
                break
            if self.handle_line(line):
                break
        self.print(f"Spice session closed: {self.result.session_id}")
        self.print(f"turns: {self.result.turns}")
        return self.result

    def handle_line(self, raw_line: str) -> bool:
        lines = [line.strip() for line in raw_line.splitlines() if line.strip()]
        if not lines:
            return False
        should_exit = False
        for line in lines:
            if line.startswith("/"):
                should_exit = self._handle_command(line)
            elif self.pending_decision is not None:
                self._handle_decision_feedback(line)
                should_exit = False
            elif self._handle_continuation(line):
                should_exit = False
            else:
                self._run_intent(line, mode=self.run_intent_mode, full_loop_preview=self.full_loop_preview)
                should_exit = False
            if should_exit:
                break
        return should_exit

    def print(self, renderable: Any) -> None:
        if isinstance(renderable, str):
            stream = self.output_stream
            if stream is None and self.console is not None:
                stream = getattr(self.console, "file", None)
            if stream is None:
                print(renderable)
            else:
                stream.write(f"{renderable}\n")
                stream.flush()
            return
        if self.console is not None:
            self.console.print(renderable)
            return
        if not isinstance(renderable, str):
            try:
                from rich.console import Console

                Console(file=self.output_stream, force_terminal=None, legacy_windows=False).print(renderable)
                return
            except ImportError:
                pass
        stream = self.output_stream
        if stream is None:
            print(str(renderable))
        else:
            stream.write(f"{renderable}\n")
            stream.flush()

    def _render_width(self) -> int:
        if self.console is not None:
            try:
                return int(self.console.size.width)
            except (AttributeError, TypeError, ValueError):
                pass
        return shutil.get_terminal_size((100, 24)).columns

    def render_help(self, *, compact: bool = False) -> Any:
        text = "\n".join(
            [
                "Commands:",
                "- type any intent to run the default decision loop",
                "- /act <intent>       run an execution-handoff decision",
                "- /advise <intent>    run a decision-only advisory turn",
                "- /refine <feedback>  refine the latest decision card",
                "- /perceive [opts]    pull external signals once; optionally open a Decision Card",
                "- /pending           continue a pending approval flow",
                "- /approval <id>     open an approval action menu",
                "- /approvals         list approval checkpoints",
                "- /approve <id>      approve a pending checkpoint",
                "- /reject <id> [why] reject a pending checkpoint",
                "- /details <id>      show approval details",
                "- /execute <id>      execute using configured executor",
                "- /dry-run <id>      run the local dry-run executor bridge",
                "- /session           show current session summary",
                "- /timeline          show current session timeline",
                "- /stats             show session stats",
                "- /doctor            check workspace health",
                "- /context [--json]  show the compiled model context",
                "- /state             show General Decision state",
                "- /refresh           redraw the startup banner",
                "- /help              show this help",
                "- /exit              close the shell",
            ]
        )
        if compact:
            text = "\n".join(text.splitlines()[:6] + ["- /help              show all commands"])
        try:
            from rich import box
            from rich.panel import Panel
        except ImportError:
            return text
        return Panel(
            text,
            title="[bold red]HELP[/bold red]",
            border_style=SpiceTheme.PANEL_BORDER,
            box=box.ROUNDED,
        )

    def _handle_command(self, line: str) -> bool:
        command, value = _split_command(line)
        if command in {"/exit", "/quit"}:
            return True
        if command == "/help":
            self.print(self.render_help())
            return False
        if command == "/refresh":
            self._refresh_banner()
            return False
        if command in {"/refine", "/explore"}:
            if command == "/explore":
                self.print("explore is not available yet; use /refine to update the latest decision card.")
                return False
            if not value:
                self.print("error: /refine requires feedback text")
                return False
            self._refine_decision(value)
            return False
        if command == "/doctor":
            self.print(render_doctor_panel(run_doctor(self.project_root)))
            return False
        if command == "/state":
            self._show_state()
            return False
        if command == "/context":
            self._show_context(value)
            return False
        if command == "/perceive":
            self._perceive(value)
            return False
        if command == "/session":
            self._show_session()
            return False
        if command == "/timeline":
            self._show_timeline()
            return False
        if command in {"/stats", "/metrics"}:
            self._show_stats()
            return False
        if command == "/approvals":
            self._show_approvals()
            return False
        if command == "/pending":
            self._open_pending_approval_flow()
            return False
        if command == "/approval":
            if not value:
                self.print("error: /approval requires an approval id")
                return False
            self._open_pending_approval_flow(value)
            return False
        if command in {"/details", "/show"}:
            approval_id = self._single_approval_id(command, value)
            if approval_id:
                self._show_approval_details(approval_id)
            return False
        if command in {"/approve", "/yes", "/y"}:
            approval_id = self._single_approval_id(command, value)
            if approval_id:
                self._resolve_approval(approval_id, status="approved")
            return False
        if command in {"/reject", "/no", "/n"}:
            approval_id, reason = _split_first(value)
            self._resolve_approval(approval_id, status="rejected", reason=reason)
            return False
        if command == "/dry-run":
            approval_id = self._single_approval_id(command, value)
            if approval_id:
                self._execute_dry_run(approval_id)
            return False
        if command == "/execute":
            approval_id = self._single_approval_id(command, value)
            if approval_id:
                self._execute_configured(approval_id)
            return False
        if command == "/act":
            if not value:
                self.print("error: /act requires an intent")
                return False
            self._run_intent(value, mode="act", full_loop_preview=self.full_loop_preview)
            return False
        if command == "/advise":
            if not value:
                self.print("error: /advise requires an intent")
                return False
            self._run_intent(value, mode="advise", full_loop_preview=False)
            return False
        self.print(f"unknown command: {command}. Type /help for commands.")
        return False

    def _render_startup_banner(self, *, config: dict[str, Any], session_payload: dict[str, Any]) -> None:
        if self.store is None:
            return
        self.print(
            render_banner(
                config,
                session_payload,
                dashboard=_startup_dashboard(self.store, config),
                width=self._render_width(),
            )
        )

    def _refresh_banner(self) -> None:
        if self.store is None:
            self.print("error: shell store is not initialized")
            return
        config = load_workspace_config(self.project_root).to_payload()
        session = load_or_create_session(self.store, session_id=self.result.session_id)
        self._render_startup_banner(config=config, session_payload=session.to_payload())

    def _single_approval_id(self, command: str, value: str) -> str:
        tokens = value.split()
        if len(tokens) != 1:
            self.print(f"error: {command} requires exactly one approval id")
            return ""
        return tokens[0]

    def _run_intent(self, intent: str, *, mode: str, full_loop_preview: bool) -> None:
        try:
            config = load_workspace_config(self.project_root).to_payload()
            with decision_progress(self.console, config=config, mode=mode):
                result = run_once(
                    intent,
                    project_root=self.project_root,
                    session_id=self.result.session_id,
                    use_bars=self.use_bars,
                    persist=self.persist,
                    full_loop_preview=full_loop_preview,
                    run_intent_mode=mode,
                )
        except Exception as exc:
            self.print(f"error: {exc}")
            return
        self._record_run(result)
        self.print(
            render_decision_card(
                result.artifact["compare_payload"],
                use_bars=self.use_bars,
                width=self._render_width(),
            )
        )
        self.print(self._artifacts_text(result))
        self._set_pending_decision(result.artifact)

    def _refine_decision(self, refinement: str) -> None:
        try:
            config = load_workspace_config(self.project_root).to_payload()
            with decision_progress(self.console, config=config, mode="refine"):
                result = refine_decision(
                    refinement,
                    project_root=self.project_root,
                    session_id=self.result.session_id,
                    use_bars=self.use_bars,
                    persist=self.persist,
                    full_loop_preview=self.full_loop_preview,
                )
        except Exception as exc:
            self.print(f"error: {exc}")
            return
        self._record_refine(result)
        self.print(
            render_decision_card(
                result.artifact["compare_payload"],
                use_bars=self.use_bars,
                width=self._render_width(),
            )
        )
        self.print(self._refine_artifacts_text(result))
        self._set_pending_decision(result.artifact)

    def _perceive(self, value: str) -> None:
        try:
            options = _parse_perceive_args(value)
            result = perceive_once(
                project_root=self.project_root,
                provider=options.get("provider"),
                poll_url=options.get("poll_url"),
                poll_command=options.get("poll_command"),
                openchronicle_mcp_url=options.get("openchronicle_mcp_url"),
                openchronicle_since_minutes=options.get("openchronicle_since_minutes"),
                openchronicle_context_limit=options.get("openchronicle_context_limit"),
                allow_command_poll=options.get("allow_command_poll"),
                decide_on_change=options.get("decide_on_change"),
                timeout_seconds=options.get("timeout_seconds"),
            )
        except Exception as exc:
            self.print(f"error: {exc}")
            return
        artifact = result.artifact
        run_id = str(artifact.get("run_id") or "")
        if run_id:
            self.result.run_ids.append(run_id)
        self.result.turns += 1
        self.print(render_perception_panel(artifact))
        self.print(
            "\n".join(
                [
                    "Perception artifacts:",
                    f"  perception={result.perception_path}",
                    f"  state={result.state_path}",
                ]
            )
        )

    def _show_state(self) -> None:
        try:
            self.print(render_state_panel(self._store().load_state()))
        except Exception as exc:
            self.print(f"error: {exc}")

    def _show_context(self, value: str = "") -> None:
        try:
            context = self._compile_debug_decision_context()
            payload = payload_value(context)
            if _context_json_requested(value):
                self.print(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2))
                return
            self.print(_render_context_panel(payload))
        except Exception as exc:
            self.print(f"error: {exc}")

    def _show_session(self) -> None:
        try:
            session = load_or_create_session(self._store(), session_id=self.result.session_id)
            self.print(render_session_panel(session))
        except Exception as exc:
            self.print(f"error: {exc}")

    def _show_timeline(self) -> None:
        try:
            session = load_or_create_session(self._store(), session_id=self.result.session_id)
            self.print(render_timeline_panel(build_session_timeline(self._store(), session), session_id=session.session_id))
        except Exception as exc:
            self.print(f"error: {exc}")

    def _show_stats(self) -> None:
        try:
            self.print(render_stats_panel(session_stats(self._store())))
        except Exception as exc:
            self.print(f"error: {exc}")

    def _show_approvals(self) -> None:
        try:
            self.print(render_approvals_panel(list_approvals(self._store())))
        except Exception as exc:
            self.print(f"error: {exc}")

    def _open_pending_approval_flow(self, approval_id: str | None = None) -> None:
        if approval_id:
            try:
                approval = load_approval(self._store(), approval_id)
            except Exception as exc:
                self.print(f"error: {exc}")
                return
            if approval.status != "pending":
                self.print(f"Approval {approval_id} is not pending. Current status: {approval.status}.")
                if approval.status == "approved":
                    self.print(f"Run /execute {approval_id} to execute, or /details {approval_id} to inspect it.")
                return
            self._approval_action_menu(approval_id)
            return

        try:
            approvals = self._pending_approvals()
        except Exception as exc:
            self.print(f"error: {exc}")
            return
        if not approvals:
            self.print("No pending approvals.")
            return
        if len(approvals) == 1:
            selected = approvals[0].approval_id
            self.print(f"Pending approval: {selected}")
            self._approval_action_menu(selected)
            return
        selected = self._choose_pending_approval(approvals)
        if selected:
            self._approval_action_menu(selected)

    def _pending_approvals(self) -> list[Any]:
        return list_approvals(self._store(), status="pending")

    def _choose_pending_approval(self, approvals: list[Any]) -> str | None:
        if self._decision_action_picker_available():
            return self._prompt_pending_approval_choice(approvals)
        self.print(_pending_approval_choice_text(approvals))
        return None

    def _prompt_pending_approval_choice(self, approvals: list[Any]) -> str | None:
        rows = [
            (str(index), approval.approval_id, _approval_summary(approval))
            for index, approval in enumerate(approvals, start=1)
        ]
        try:
            raw = self._prompt_inline_choice(
                title="Which pending approval?",
                rows=rows,
                prompt_label="pending",
            )
        except Exception:
            return None
        value = str(raw or "").strip()
        if not value:
            return None
        if value.isdigit():
            index = int(value) - 1
            if 0 <= index < len(approvals):
                return str(approvals[index].approval_id)
        for approval in approvals:
            if value == approval.approval_id:
                return str(approval.approval_id)
        self.print(f"Unknown pending approval selection: {value}")
        return None

    def _approval_action_menu(self, approval_id: str) -> None:
        while True:
            if self._decision_action_picker_available():
                action = self._prompt_decision_action({"approval_id": approval_id})
                if not action:
                    self.print(_decision_next_step_text({"approval_id": approval_id}))
                    return
            else:
                self.print(_decision_next_step_text({"approval_id": approval_id}))
                return
            if self._handle_approval_action(action, approval_id):
                return

    def _show_approval_details(self, approval_id: str) -> None:
        if not approval_id:
            self.print("error: approval id required")
            return
        try:
            self.print(render_approval_details_panel(load_approval(self._store(), approval_id)))
        except Exception as exc:
            self.print(f"error: {exc}")

    def _resolve_approval(self, approval_id: str, *, status: str, reason: str = "") -> bool:
        if not approval_id:
            self.print("error: approval id required")
            return False
        try:
            if status == "approved":
                resolved = approve_approval(self._store(), approval_id)
                self.result.approved_ids.append(approval_id)
            else:
                resolved = reject_approval(self._store(), approval_id, reason=reason)
                self.result.rejected_ids.append(approval_id)
            self.print(render_approval_resolution_panel(resolved))
            self._clear_pending_decision(approval_id)
            return True
        except Exception as exc:
            self.print(f"error: {exc}")
            return False

    def _execute_dry_run(self, approval_id: str) -> None:
        if not approval_id:
            self.print("error: approval id required")
            return
        try:
            execution = execute_dry_run_approval(approval_id, project_root=self.project_root)
            outcome_id = str(execution.artifact.get("outcome_id") or "")
            if outcome_id:
                self.result.dry_run_outcome_ids.append(outcome_id)
            self.print(render_execution_panel(execution.artifact, execution.rendered_text))
        except Exception as exc:
            self.print(f"error: {exc}")

    def _execute_configured(self, approval_id: str, *, permission_mode: str | None = None) -> None:
        if not approval_id:
            self.print("error: approval id required")
            return
        try:
            config = load_workspace_config(self.project_root)
            executor = (
                resolve_executor_runtime_from_config_with_permission(config, permission_mode)
                if permission_mode
                else resolve_executor_runtime_from_config(config)
            )
            if permission_mode is None and executor.permission_enforcement == "command_flag":
                permission_mode = self._confirm_executor_permission_for_approval(approval_id)
                if permission_mode is None:
                    return
                executor = resolve_executor_runtime_from_config_with_permission(
                    config,
                    permission_mode,
                )
            if executor.status == "unsupported":
                raise ValueError(executor.detail)
            if executor.status != "ready":
                raise ValueError(executor.detail)
            self.print(
                render_execution_dispatch_panel(
                    approval_id=approval_id,
                    executor_provider=executor.executor_id,
                    executor_command=executor.command,
                )
            )
            with execution_progress(self.console, approval_id=approval_id, executor=executor):
                execution = self._run_configured_executor(approval_id, executor)
            outcome_id = str(execution.artifact.get("outcome_id") or "")
            if outcome_id:
                self.result.dry_run_outcome_ids.append(outcome_id)
            self.print(render_execution_summary_panel(execution.artifact))
        except Exception as exc:
            provider = "executor"
            try:
                provider = resolve_executor_runtime_from_config(load_workspace_config(self.project_root)).executor_id
            except Exception:
                pass
            self.print(render_execution_error_panel(approval_id=approval_id, executor_provider=provider, error=exc))

    def _run_configured_executor(self, approval_id: str, executor: Any) -> Any:
        if executor.executor_id == "dry_run":
            return execute_dry_run_approval(approval_id, project_root=self.project_root)
        if executor.executor_id == "sdep_subprocess":
            return execute_sdep_subprocess_approval(
                approval_id,
                command=executor.command,
                project_root=self.project_root,
            )
        if executor.executor_id == "codex":
            return execute_codex_approval(
                approval_id,
                command=executor.command,
                project_root=self.project_root,
            )
        if executor.executor_id == "claude_code":
            return execute_claude_code_approval(
                approval_id,
                command=executor.command,
                project_root=self.project_root,
            )
        if executor.executor_id == "hermes":
            return execute_hermes_approval(
                approval_id,
                command=executor.command,
                project_root=self.project_root,
            )
        raise ValueError(f"Unsupported executor in .spice/config.json: {executor.executor_id!r}.")

    def _record_run(self, result: RunOnceResult) -> None:
        run_id = str(result.artifact.get("run_id") or "")
        if run_id:
            self.result.run_ids.append(run_id)
        self.result.turns += 1

    def _record_refine(self, result: RefineResult) -> None:
        run_id = str(result.artifact.get("run_id") or "")
        if run_id:
            self.result.run_ids.append(run_id)
        self.result.turns += 1

    def _artifacts_text(self, result: RunOnceResult) -> str:
        lines = [
            "Artifacts:",
            f"  run={result.run_path}",
            f"  decision={result.decision_path}",
        ]
        if result.approval_path is not None:
            lines.append(f"  approval={result.approval_path}")
        lines.append(f"  session={result.session_path}")
        lines.append(f"  state={result.state_path}")
        return "\n".join(lines)

    def _refine_artifacts_text(self, result: RefineResult) -> str:
        lines = [
            "Refine artifacts:",
            f"  run={result.run_path}",
            f"  decision={result.decision_path}",
        ]
        if result.approval_path is not None:
            lines.append(f"  approval={result.approval_path}")
        lines.append(f"  session={result.session_path}")
        lines.append(f"  state={result.state_path}")
        return "\n".join(lines)

    def _store(self) -> LocalJsonStore:
        if self.store is None:
            self.store = LocalJsonStore(require_workspace(self.project_root))
        return self.store

    def _prompt_text(self) -> str:
        return "decision> " if self.pending_decision is not None else "spice> "

    def _set_pending_decision(self, artifact: dict[str, Any]) -> None:
        approval_id = str(artifact.get("approval_id") or "").strip()
        if not approval_id:
            self.pending_decision = None
            return
        self.pending_decision = {
            "approval_id": approval_id,
            "run_id": str(artifact.get("run_id") or ""),
            "decision_id": str(artifact.get("decision_id") or ""),
            "candidate_id": str(artifact.get("selected_candidate_id") or ""),
        }
        if self._decision_action_picker_available():
            self._show_decision_action_picker()
            return
        self.print(_decision_next_step_text(self.pending_decision))

    def _clear_pending_decision(self, approval_id: str | None = None) -> None:
        if self.pending_decision is None:
            return
        if approval_id is None or self.pending_decision.get("approval_id") == approval_id:
            self.pending_decision = None

    def _handle_decision_feedback(self, line: str) -> None:
        text = line.strip()
        if not text:
            return
        approval_id = str((self.pending_decision or {}).get("approval_id") or "")
        command, value = _split_first(text)
        normalized = command.lower()
        if normalized in {"y", "yes"}:
            self._approve_and_execute_pending(approval_id)
            return
        if normalized in {"a", "approve"}:
            self._resolve_approval(approval_id, status="approved")
            return
        if normalized in {"n", "no", "reject"}:
            self._resolve_approval(approval_id, status="rejected", reason=value)
            return
        if normalized in {"d", "detail", "details"}:
            self._show_approval_details(approval_id)
            return
        if normalized in {"q", "skip"}:
            self.print(f"Skipped pending decision: {approval_id}")
            self._clear_pending_decision(approval_id)
            return
        if normalized in {"r", "refine"} and value:
            self._refine_decision(value)
            return
        if normalized in {"r", "refine"}:
            self.print("error: refine requires feedback text")
            return
        resolution = self._resolve_continuation(text)
        if resolution.is_continuation:
            self._handle_continuation_resolution(resolution)
            return
        self._refine_decision(text)

    def _handle_continuation(self, line: str) -> bool:
        resolution = self._resolve_continuation(line)
        if not resolution.is_continuation:
            return False
        self._handle_continuation_resolution(resolution)
        return True

    def _resolve_continuation(self, line: str) -> ContinuationResolution:
        try:
            config = load_workspace_config(self.project_root).to_payload()
        except Exception:
            config = {}
        return resolve_continuation_from_runtime_config(
            line,
            self._active_decision_frame(),
            config=config,
        )

    def _handle_continuation_resolution(self, resolution: ContinuationResolution) -> None:
        action = resolution.action
        if action == "choose_option":
            self._choose_active_frame_option(resolution)
            return
        if action == "execute_selected":
            self._execute_active_frame_selected()
            return
        if action == "approve_execute":
            approval_id = self._active_frame_approval_id()
            if approval_id:
                self._approve_and_execute_pending(approval_id)
                return
            self.print("No approval is attached to the current Decision Card. Type `execute selected` to open an execution handoff.")
            return
        if action == "approve_only":
            approval_id = self._active_frame_approval_id()
            if approval_id:
                self._resolve_approval(approval_id, status="approved")
                return
            self.print("No approval is attached to the current Decision Card. Type `execute selected` to open an execution handoff.")
            return
        if action == "show_details":
            approval_id = self._active_frame_approval_id()
            if approval_id:
                self._show_approval_details(approval_id)
            else:
                self._show_active_frame_details()
            return
        if action == "skip":
            self._mark_active_frame_status("skipped")
            self.print("Skipped current Decision Card.")
            self.pending_decision = None
            return
        if action == "refine":
            self._refine_decision(resolution.text)
            return
        self.print(f"Could not continue from current Decision Card: {resolution.text}")

    def _choose_active_frame_option(self, resolution: ContinuationResolution) -> None:
        frame = self._active_decision_frame()
        if not frame:
            self.print("No active Decision Card to choose from.")
            return
        updated = update_frame_selected_candidate(frame, resolution.candidate_id)
        if updated == frame:
            self.print(f"Could not find option {resolution.label}.")
            return
        self._save_active_decision_frame(updated)
        selected = updated.get("selected") if isinstance(updated.get("selected"), dict) else {}
        self.pending_decision = None
        title = str(selected.get("title") or selected.get("recommended_action") or resolution.label)
        self.print(
            "\n".join(
                [
                    f"Selected {selected.get('label') or resolution.label}: {title}",
                    f"candidate_id: {selected.get('candidate_id') or resolution.candidate_id}",
                    "Next: type `execute selected`, `refine that ...`, `details`, or a new intent.",
                ]
            )
        )

    def _execute_active_frame_selected(self) -> None:
        approval_id = self._active_frame_approval_id()
        if approval_id:
            self._approve_and_execute_pending(approval_id)
            return
        frame = self._active_decision_frame()
        if not frame:
            self.print("No active Decision Card to execute.")
            return
        intent = selected_candidate_execution_text(frame)
        if not intent:
            self.print("The selected candidate has no executor task to run.")
            return
        self._run_intent(intent, mode="act", full_loop_preview=self.full_loop_preview)

    def _active_frame_approval_id(self) -> str:
        if self.pending_decision is not None:
            approval_id = str(self.pending_decision.get("approval_id") or "").strip()
            if approval_id:
                return approval_id
        frame = self._active_decision_frame()
        return str(frame.get("approval_id") or "").strip() if frame else ""

    def _active_decision_frame(self) -> dict[str, Any]:
        try:
            payload = self._store().load_state()
        except Exception:
            return {}
        general = _general_state_payload(payload)
        metadata = general.get("metadata")
        if not isinstance(metadata, dict):
            return {}
        frame = metadata.get("active_decision_frame")
        return dict(frame) if isinstance(frame, dict) else {}

    def _compile_debug_decision_context(self) -> Any:
        store = self._store()
        config = load_workspace_config(self.project_root)
        config_payload = config.to_payload()
        state_payload = store.load_state()
        world_state = _world_state_from_workspace_payload(state_payload)
        general_state = load_general_state(world_state)
        session = load_or_create_session(store, session_id=self.result.session_id)
        frame = self._active_decision_frame()
        compiler = load_workspace_context_compiler(
            self.project_root,
            config=config,
        )
        return compiler.compile_general_decision_context(
            world_state,
            general_state,
            current_intent=_context_current_intent(frame),
            active_decision_frame=frame,
            session=payload_value(session),
            config=config_payload,
            domain="general",
        )

    def _save_active_decision_frame(self, frame: dict[str, Any]) -> None:
        if not self.persist:
            return
        store = self._store()
        payload = store.load_state()
        general = _general_state_payload(payload)
        metadata = general.get("metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            general["metadata"] = metadata
        metadata["active_decision_frame"] = frame
        store.save_state(payload)

    def _mark_active_frame_status(self, status: str) -> None:
        frame = self._active_decision_frame()
        if not frame:
            return
        frame["status"] = status
        self._save_active_decision_frame(frame)

    def _show_active_frame_details(self) -> None:
        frame = self._active_decision_frame()
        if not frame:
            self.print("No active Decision Card.")
            return
        selected = frame.get("selected") if isinstance(frame.get("selected"), dict) else {}
        self.print(
            "\n".join(
                [
                    "ACTIVE DECISION FRAME",
                    f"decision_id: {frame.get('decision_id')}",
                    f"selected: {selected.get('label') or ''} {selected.get('title') or ''}".rstrip(),
                    f"candidate_id: {selected.get('candidate_id') or frame.get('selected_candidate_id') or ''}",
                    f"status: {frame.get('status') or 'unknown'}",
                ]
            )
        )

    def _decision_action_picker_available(self) -> bool:
        if self.output_stream is None:
            is_tty = sys.stdin.isatty() and sys.stdout.isatty()
        else:
            is_tty = bool(getattr(self.output_stream, "isatty", lambda: False)())
        return is_tty and _prompt_toolkit_available()

    def _show_decision_action_picker(self) -> None:
        while self.pending_decision is not None:
            action = self._prompt_decision_action(self.pending_decision)
            if not action:
                self.print(_decision_next_step_text(self.pending_decision))
                return
            if self._handle_decision_action(action):
                return

    def _prompt_decision_action(self, pending: dict[str, str]) -> str:
        options = _decision_action_options()
        rows = [(value, label, shortcut) for value, label, shortcut in options]
        try:
            result = self._prompt_inline_choice(
                title="What would you like to do?",
                rows=rows,
                footer=f"approval_id: {pending.get('approval_id', '')}",
                prompt_label="action",
            )
        except (EOFError, KeyboardInterrupt):
            return ""
        except Exception:
            return ""
        return _normalize_decision_action(str(result or ""))

    def _prompt_inline_choice(
        self,
        *,
        title: str,
        rows: list[tuple[str, str, str]],
        prompt_label: str,
        footer: str = "",
    ) -> str:
        try:
            from prompt_toolkit.application import Application
            from prompt_toolkit.formatted_text import FormattedText
            from prompt_toolkit.key_binding import KeyBindings
            from prompt_toolkit.layout import HSplit, Layout, Window
            from prompt_toolkit.layout.controls import FormattedTextControl
            from prompt_toolkit.styles import Style
        except ImportError:
            return ""

        selected = {"index": 0}
        bindings = KeyBindings()

        @bindings.add("up")
        def _move_up(event: Any) -> None:
            selected["index"] = (selected["index"] - 1) % len(rows)
            event.app.invalidate()

        @bindings.add("down")
        def _move_down(event: Any) -> None:
            selected["index"] = (selected["index"] + 1) % len(rows)
            event.app.invalidate()

        @bindings.add("enter")
        def _accept(event: Any) -> None:
            event.app.exit(result=rows[selected["index"]][0])

        @bindings.add("space")
        def _accept_space(event: Any) -> None:
            event.app.exit(result=rows[selected["index"]][0])

        @bindings.add("escape")
        def _cancel(event: Any) -> None:
            event.app.exit(result="")

        control = FormattedTextControl(
            lambda: FormattedText(
                _inline_choice_fragments(
                    title=title,
                    rows=rows,
                    selected_index=selected["index"],
                    footer=footer,
                    prompt_label=prompt_label,
                )
            ),
            focusable=True,
        )
        app = Application(
            layout=Layout(HSplit([Window(content=control, dont_extend_height=True)]), focused_element=control),
            key_bindings=bindings,
            style=Style.from_dict(
                {
                    "title": "ansiyellow bold",
                    "shortcut": "ansigreen",
                    "selected": "ansigreen bold",
                    "choice": "ansigreen",
                    "hint": "ansibrightblack",
                    "prompt": "ansired bold",
                    "cursor": "ansigreen bold",
                }
            ),
            full_screen=False,
            mouse_support=False,
        )
        return str(app.run() or "")

    def _handle_decision_action(self, action: str) -> bool:
        approval_id = str((self.pending_decision or {}).get("approval_id") or "")
        return self._handle_approval_action(action, approval_id)

    def _handle_approval_action(self, action: str, approval_id: str) -> bool:
        if action == "approve_execute":
            self._approve_and_execute_pending(approval_id)
            return True
        if action == "approve":
            self._resolve_approval(approval_id, status="approved")
            return True
        if action == "reject":
            reason = self._prompt_free_text("reject reason> ")
            if reason is None:
                self.print("Reject cancelled.")
                return False
            self._resolve_approval(approval_id, status="rejected", reason=reason)
            return True
        if action == "refine":
            feedback = self._prompt_free_text("refine> ")
            if not feedback:
                self.print("Refine cancelled.")
                return False
            self._refine_decision(feedback)
            return True
        if action == "details":
            self._show_approval_details(approval_id)
            return False
        if action == "skip":
            self.print(f"Skipped pending decision: {approval_id}")
            self._clear_pending_decision(approval_id)
            return True
        return False

    def _prompt_free_text(self, prompt: str) -> str | None:
        try:
            return str(self.prompt_session.prompt([("class:prompt", prompt)])).strip()
        except (EOFError, KeyboardInterrupt):
            return None

    def _approve_and_execute_pending(self, approval_id: str) -> None:
        if not approval_id:
            self.print("error: no pending approval to execute")
            self.pending_decision = None
            return
        permission_mode = self._confirm_executor_permission_for_approval(approval_id)
        if permission_mode is None:
            return
        if self._resolve_approval(approval_id, status="approved"):
            self._execute_configured(approval_id, permission_mode=permission_mode)

    def _confirm_executor_permission_for_approval(self, approval_id: str) -> str | None:
        store = self._store()
        approval = load_approval(store, approval_id)
        config = load_workspace_config(self.project_root)
        executor = resolve_executor_runtime_from_config(config)
        escalate, requirement = approval_requires_permission_escalation(
            store=store,
            approval=approval,
            current_permission=executor.permission_mode,
        )
        if not escalate:
            return executor.permission_mode
        if executor.permission_enforcement != "command_flag":
            self.print(
                "This approval requires "
                f"{requirement.required_permission}, but {executor.executor_id} permission escalation "
                "is not automated yet. The approval remains pending."
            )
            return None
        while True:
            action = self._prompt_permission_escalation(
                approval_id=approval_id,
                executor_id=executor.executor_id,
                current_permission=executor.permission_mode,
                required_permission=requirement.required_permission,
                reason=requirement.reason,
            )
            if action == "yes":
                return requirement.required_permission
            if action == "details":
                self.print(
                    _permission_escalation_details(
                        approval_id=approval_id,
                        executor_id=executor.executor_id,
                        current_permission=executor.permission_mode,
                        required_permission=requirement.required_permission,
                        reason=requirement.reason,
                    )
                )
                continue
            self.print(
                "Execution permission was not escalated. "
                f"Approval remains pending: {approval_id}"
            )
            return None

    def _prompt_permission_escalation(
        self,
        *,
        approval_id: str,
        executor_id: str,
        current_permission: str,
        required_permission: str,
        reason: str,
    ) -> str:
        rows = [
            ("yes", f"Yes, run {executor_id} with {required_permission} for this execution", "y / yes"),
            ("no", "No, keep this approval pending", "n / no"),
            ("details", "Show permission details", "d / details"),
        ]
        footer = (
            f"approval_id: {approval_id}\n"
            f"current: {current_permission}  required: {required_permission}\n"
            f"reason: {reason}"
        )
        if self._decision_action_picker_available():
            try:
                value = self._prompt_inline_choice(
                    title="Higher execution permission required",
                    rows=rows,
                    footer=footer,
                    prompt_label="permission",
                )
                normalized = str(value or "").strip().lower()
                if normalized in {"yes", "no", "details"}:
                    return normalized
            except Exception:
                pass
        self.print(_permission_escalation_text(footer))
        answer = self._prompt_free_text("permission> ")
        normalized = str(answer or "").strip().lower()
        if normalized in {"y", "yes"}:
            return "yes"
        if normalized in {"d", "detail", "details"}:
            return "details"
        return "no"

    def _build_console(self, output_stream: TextIO | None) -> Any:
        try:
            from rich.console import Console
        except ImportError:
            return None
        if output_stream is None:
            return Console(force_terminal=None, legacy_windows=False)
        return Console(file=output_stream, force_terminal=None, legacy_windows=False)

    def _build_prompt_session(self) -> Any:
        from prompt_toolkit import PromptSession
        from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
        from prompt_toolkit.completion import WordCompleter
        from prompt_toolkit.history import FileHistory
        from prompt_toolkit.styles import Style

        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        return PromptSession(
            history=FileHistory(str(self.history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=WordCompleter(COMMANDS, ignore_case=True),
            style=Style.from_dict({"prompt": "ansired bold"}),
        )


def _prompt_toolkit_available() -> bool:
    try:
        import prompt_toolkit  # noqa: F401
    except ImportError:
        return False
    return True


def _context_json_requested(value: str) -> bool:
    tokens = [token.strip().lower() for token in shlex.split(value or "")]
    return "--json" in tokens or "json" in tokens


def _context_current_intent(frame: dict[str, Any]) -> dict[str, Any]:
    raw_input = frame.get("input") if isinstance(frame.get("input"), dict) else {}
    text = str(raw_input.get("text") or "").strip()
    return {
        "text": text,
        "source": str(frame.get("source") or "context_debug"),
        "kind": "context_debug",
        "run_intent_mode": str(frame.get("run_intent_mode") or ""),
        "display_language": str(frame.get("display_language") or ""),
        "decision_id": str(frame.get("decision_id") or ""),
        "run_id": str(frame.get("run_id") or ""),
    }


def _render_context_panel(payload: dict[str, Any]) -> Any:
    try:
        from rich import box
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        return _context_text(payload)

    frame = payload.get("active_decision_frame")
    frame_payload = frame if isinstance(frame, dict) else {}
    current_intent = payload.get("current_intent")
    intent_payload = current_intent if isinstance(current_intent, dict) else {}
    executor = payload.get("executor_affordance")
    executor_payload = executor if isinstance(executor, dict) else {}
    session = payload.get("session_summary")
    session_payload = session if isinstance(session, dict) else {}
    workspace = payload.get("workspace_context")
    workspace_payload = workspace if isinstance(workspace, dict) else {}

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("field", style=SpiceTheme.DIM, no_wrap=True)
    table.add_column("value")
    table.add_row("context_id", str(payload.get("id") or ""))
    table.add_row("context_type", str(payload.get("context_type") or ""))
    table.add_row("current_intent", _shorten(str(intent_payload.get("text") or ""), 120))
    table.add_row("active_decision", str(frame_payload.get("decision_id") or ""))
    table.add_row("selected", _context_selected_summary(frame_payload))
    table.add_row("recent_decisions", str(len(_list(payload.get("recent_decisions")))))
    table.add_row("recent_approvals", str(len(_list(payload.get("recent_approvals")))))
    table.add_row("recent_outcomes", str(len(_list(payload.get("recent_outcomes")))))
    table.add_row("retrieved_memory", str(len(_list(payload.get("retrieved_memory")))))
    table.add_row("executor", _executor_context_summary(executor_payload))
    table.add_row("summary", _summary_context_summary(session_payload))
    table.add_row("session", _session_context_summary(session_payload))
    table.add_row("workspace", _workspace_context_summary(workspace_payload))
    table.add_row("refs", str(len(_list(payload.get("refs")))))
    return Panel(
        table,
        title="[bold red]COMPILED DECISION CONTEXT[/bold red]",
        subtitle="Use /context --json to inspect the exact payload.",
        border_style=SpiceTheme.PANEL_BORDER,
        box=box.ROUNDED,
    )


def _context_text(payload: dict[str, Any]) -> str:
    return "\n".join(
        [
            "COMPILED DECISION CONTEXT",
            f"context_id: {payload.get('id') or ''}",
            f"context_type: {payload.get('context_type') or ''}",
            f"current_intent: {_shorten(str(_mapping(payload.get('current_intent')).get('text') or ''), 120)}",
            f"active_decision: {_mapping(payload.get('active_decision_frame')).get('decision_id') or ''}",
            f"selected: {_context_selected_summary(_mapping(payload.get('active_decision_frame')))}",
            f"recent_decisions: {len(_list(payload.get('recent_decisions')))}",
            f"recent_approvals: {len(_list(payload.get('recent_approvals')))}",
            f"recent_outcomes: {len(_list(payload.get('recent_outcomes')))}",
            f"retrieved_memory: {len(_list(payload.get('retrieved_memory')))}",
            f"executor: {_executor_context_summary(_mapping(payload.get('executor_affordance')))}",
            f"summary: {_summary_context_summary(_mapping(payload.get('session_summary')))}",
            f"session: {_session_context_summary(_mapping(payload.get('session_summary')))}",
            f"workspace: {_workspace_context_summary(_mapping(payload.get('workspace_context')))}",
            "Use /context --json to inspect the exact payload.",
        ]
    )


def _context_selected_summary(frame: dict[str, Any]) -> str:
    selected = frame.get("selected") if isinstance(frame.get("selected"), dict) else {}
    label = str(selected.get("label") or "").strip()
    title = str(selected.get("title") or selected.get("recommended_action") or "").strip()
    candidate_id = str(selected.get("candidate_id") or frame.get("selected_candidate_id") or "").strip()
    parts = [part for part in [label, _shorten(title, 80), candidate_id] if part]
    return " | ".join(parts)


def _executor_context_summary(payload: dict[str, Any]) -> str:
    parts = [
        str(payload.get("executor") or payload.get("provider") or ""),
        str(payload.get("permission") or payload.get("permission_mode") or ""),
    ]
    available = payload.get("available")
    if available is not None:
        parts.append(f"available={str(bool(available)).lower()}")
    return " ".join(part for part in parts if part).strip()


def _session_context_summary(payload: dict[str, Any]) -> str:
    session_id = str(payload.get("session_id") or "")
    runs = payload.get("runs")
    decisions = payload.get("decisions")
    parts = [session_id]
    if runs is not None:
        parts.append(f"runs={runs}")
    if decisions is not None:
        parts.append(f"decisions={decisions}")
    return " ".join(part for part in parts if part).strip()


def _summary_context_summary(payload: dict[str, Any]) -> str:
    rolling = payload.get("rolling_summary")
    if not isinstance(rolling, dict):
        return "none"
    summary_type = str(rolling.get("summary_type") or "deterministic")
    updated = str(rolling.get("updated_at") or "")
    model = rolling.get("model") if isinstance(rolling.get("model"), dict) else {}
    model_id = str(model.get("model_id") or "")
    parts = [summary_type]
    if model_id:
        parts.append(model_id)
    if updated:
        parts.append(f"updated={updated}")
    return " ".join(parts)


def _workspace_context_summary(payload: dict[str, Any]) -> str:
    parts = [
        f"memory={payload.get('memory_provider')}"
        if payload.get("memory_provider") is not None
        else "",
        f"compiler={payload.get('context_compiler')}"
        if payload.get("context_compiler") is not None
        else "",
        f"summary={payload.get('memory_summary_provider')}"
        if payload.get("memory_summary_provider") is not None
        else "",
        f"executor={payload.get('executor')}" if payload.get("executor") is not None else "",
    ]
    return " ".join(part for part in parts if part)


def _world_state_from_workspace_payload(payload: dict[str, Any]) -> WorldState:
    world_payload = payload.get("world_state")
    if not isinstance(world_payload, dict):
        raise ValueError("Workspace state must contain a world_state object.")
    return WorldState(
        id=str(world_payload.get("id") or "worldstate.local"),
        schema_version=str(world_payload.get("schema_version", "0.1")),
        status=str(world_payload.get("status", "current")),
        entities=_mapping(world_payload.get("entities")),
        relations=_list_of_mappings(world_payload.get("relations")),
        goals=_list_of_mappings(world_payload.get("goals")),
        constraints=_list_of_mappings(world_payload.get("constraints")),
        resources=_mapping(world_payload.get("resources")),
        risks=_list_of_mappings(world_payload.get("risks")),
        signals=_list_of_mappings(world_payload.get("signals")),
        active_intents=_list_of_mappings(world_payload.get("active_intents")),
        recent_outcomes=_list_of_mappings(world_payload.get("recent_outcomes")),
        confidence=_mapping(world_payload.get("confidence")),
        provenance=_mapping(world_payload.get("provenance")),
        domain_state=_mapping(world_payload.get("domain_state")),
    )


def _startup_dashboard(store: LocalJsonStore, config: dict[str, Any]) -> dict[str, Any]:
    state_payload = _safe_load_state(store)
    state_counts = _general_state_counts(state_payload)
    approval_ids = store.list_record_ids("approvals")
    pending_count = 0
    for approval_id in approval_ids:
        try:
            approval = store.load_approval(approval_id)
        except Exception:
            continue
        if approval.get("status") == "pending":
            pending_count += 1
    return {
        "mode": _runtime_mode(config),
        "pending_approvals": pending_count,
        "run_count": len(store.list_record_ids("runs")),
        "decision_count": len(store.list_record_ids("decisions")),
        "outcome_count": len(store.list_record_ids("outcomes")),
        "state_counts": state_counts,
        "executors": _executor_readiness(config),
        "skills": _skill_readiness(),
        "perception": _perception_readiness(config),
    }


def _safe_load_state(store: LocalJsonStore) -> dict[str, Any]:
    try:
        return store.load_state()
    except Exception:
        return {}


def _general_state_counts(state_payload: dict[str, Any]) -> dict[str, int]:
    general = _general_state_payload(state_payload)
    return {
        "observations": len(_items(general, "observations")),
        "intents": len(_items(general, "intents")),
        "work_items": len(_items(general, "work_items")),
        "commitments": len(_items(general, "commitments")),
        "outcomes": len(_items(general, "outcomes")),
        "approvals": len(_items(general, "approvals")),
    }


def _general_state_payload(state_payload: dict[str, Any]) -> dict[str, Any]:
    world = state_payload.get("world_state")
    if not isinstance(world, dict):
        return {}
    domain = world.get("domain_state")
    if not isinstance(domain, dict):
        return {}
    general = domain.get("general_decision")
    return general if isinstance(general, dict) else {}


def _items(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload.get(key)
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, dict) else {}


def _list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else []


def _list_of_mappings(value: Any) -> list[dict[str, Any]]:
    return [dict(item) for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _shorten(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[: max(0, limit - 1)].rstrip() + "…"


def _decision_next_step_text(pending: dict[str, str]) -> str:
    lines = [
        "Decision prompt shortcuts (type one at decision>):",
        "  y / yes                         approve and execute with the configured executor",
        "  a / approve                     approve only",
        "  n <reason> / reject <reason>    reject",
        "  r <feedback> / refine <feedback> refine and regenerate the Decision Card",
        "  d / details                     show approval details",
        "  q / skip                        skip this decision for now",
        "",
        "Examples:",
        "  decision> y",
        "  decision> reject too risky right now",
        "  decision> refine execute directly; do not split",
        f"approval_id: {pending.get('approval_id', '')}",
    ]
    return "\n".join(lines)


def _decision_action_options() -> list[tuple[str, str, str]]:
    return [
        ("approve_execute", "Approve and execute with configured executor", "y / yes"),
        ("approve", "Approve only", "a / approve"),
        ("reject", "Reject", "n / reject"),
        ("refine", "Refine with feedback", "r / refine"),
        ("details", "Show details", "d / details"),
        ("skip", "Skip for now", "q / skip"),
    ]


def _decision_action_menu_text(pending: dict[str, str], options: list[tuple[str, str, str]]) -> Any:
    try:
        from rich import box
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        lines = [
            "Decision Action",
            f"approval_id: {pending.get('approval_id', '')}",
            "Use up/down then Enter, or type a shortcut.",
        ]
        lines.extend(f"  {shortcut:<12} {label}" for _, label, shortcut in options)
        return "\n".join(lines)

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column(style=SpiceTheme.DIM, no_wrap=True)
    table.add_column(style=SpiceTheme.CANDIDATE)
    for _, label, shortcut in options:
        table.add_row(shortcut, label)
    return Panel(
        table,
        title="[bold red]DECISION ACTION[/bold red]",
        subtitle=f"approval_id: {pending.get('approval_id', '')}",
        border_style=SpiceTheme.PANEL_BORDER,
        box=box.ROUNDED,
    )


def _pending_approval_choice_text(approvals: list[Any]) -> Any:
    try:
        from rich import box
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        lines = ["Pending approvals:"]
        for index, approval in enumerate(approvals, start=1):
            lines.append(f"{index}. {approval.approval_id} - {_approval_summary(approval)}")
        lines.append("Select approval with up/down then Enter, or type a number.")
        return "\n".join(lines)

    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("#", style=SpiceTheme.DIM, no_wrap=True)
    table.add_column("approval_id", style=SpiceTheme.DIM)
    table.add_column("summary")
    for index, approval in enumerate(approvals, start=1):
        table.add_row(str(index), approval.approval_id, _approval_summary(approval))
    return Panel(
        table,
        title="[bold red]PENDING APPROVALS[/bold red]",
        subtitle="Use up/down then Enter, or type a number.",
        border_style=SpiceTheme.PANEL_BORDER,
        box=box.ROUNDED,
    )


def _inline_choice_fragments(
    *,
    title: str,
    rows: list[tuple[str, str, str]],
    selected_index: int,
    footer: str = "",
    prompt_label: str = "",
) -> list[tuple[str, str]]:
    fragments: list[tuple[str, str]] = [
        ("class:title", f"{title}\n"),
        ("class:hint", "  ↑↓ navigate   ENTER/SPACE select   ESC cancel\n\n"),
    ]
    for index, (shortcut, label, detail) in enumerate(rows):
        selected = index == selected_index
        prefix = "→ (●)" if selected else "  (○)"
        style = "class:selected" if selected else "class:choice"
        fragments.append((style, f"{prefix} {label}"))
        if detail:
            fragments.append(("class:shortcut", f"  {detail}"))
        elif shortcut:
            fragments.append(("class:shortcut", f"  {shortcut}"))
        fragments.append(("", "\n"))
    if footer:
        fragments.extend([("", "\n"), ("class:hint", f"{footer}\n")])
    if prompt_label:
        fragments.extend([("class:prompt", f"{prompt_label}> "), ("class:cursor", "█")])
    return fragments


def _approval_summary(approval: Any) -> str:
    candidate = str(getattr(approval, "candidate_id", "") or "candidate unknown")
    prompt = str(getattr(approval, "prompt", "") or "").strip()
    if prompt:
        first_line = prompt.splitlines()[0].strip()
        if first_line:
            return first_line[:80]
    return candidate[:80]


def _normalize_decision_action(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "y": "approve_execute",
        "yes": "approve_execute",
        "approve_execute": "approve_execute",
        "approve+execute": "approve_execute",
        "a": "approve",
        "approve": "approve",
        "n": "reject",
        "no": "reject",
        "reject": "reject",
        "r": "refine",
        "refine": "refine",
        "d": "details",
        "detail": "details",
        "details": "details",
        "q": "skip",
        "skip": "skip",
    }
    return aliases.get(normalized, normalized)


def _permission_escalation_text(footer: str) -> str:
    return "\n".join(
        [
            "Higher execution permission required",
            footer,
            "",
            "Choose:",
            "  y / yes      escalate and execute",
            "  n / no       keep pending",
            "  d / details  show details",
        ]
    )


def _permission_escalation_details(
    *,
    approval_id: str,
    executor_id: str,
    current_permission: str,
    required_permission: str,
    reason: str,
) -> str:
    return "\n".join(
        [
            "Execution permission details",
            f"approval_id: {approval_id}",
            f"executor: {executor_id}",
            f"current_permission: {current_permission}",
            f"required_permission: {required_permission}",
            f"reason: {reason}",
            "",
            "If approved, Spice uses the higher permission only for this execution.",
        ]
    )


def _runtime_mode(config: dict[str, Any]) -> str:
    executor = str(config.get("executor") or "dry_run")
    llm = str(config.get("llm_provider") or "deterministic")
    if executor == "dry_run" and llm == "deterministic":
        return "decision + dry-run"
    if executor == "dry_run":
        return "LLM decision + dry-run"
    return "configured executor handoff"


def _executor_readiness(config: dict[str, Any]) -> list[dict[str, str]]:
    try:
        configured_config = SpiceWorkspaceConfig.from_payload(config)
    except Exception:
        configured_config = SpiceWorkspaceConfig()
    configured = str(config.get("executor") or "dry_run")
    items: list[dict[str, str]] = []
    for executor in ["dry_run", "sdep_subprocess", "codex", "claude_code", "hermes"]:
        if executor == "dry_run":
            status = "ready"
        elif configured == executor:
            runtime = resolve_executor_runtime_from_config(configured_config)
            status = "configured" if runtime.status == "ready" else runtime.detail
        else:
            status = "not_configured"
        items.append({"name": executor, "status": status})
    return items


def _skill_readiness() -> list[dict[str, str]]:
    return [
        {"name": "item.triage", "status": "ready"},
        {"name": "artifact.draft", "status": "ready"},
        {"name": "task.split", "status": "ready"},
        {"name": "intent.execute", "status": "ready"},
        {"name": "capability.use", "status": "ready"},
    ]


def _perception_readiness(config: dict[str, Any]) -> list[dict[str, str]]:
    configured = str(config.get("perception_provider") or "manual")
    poll_ready = bool(
        str(config.get("perception_poll_url") or "").strip()
        or str(config.get("perception_poll_command") or "").strip()
    )
    open_chronicle_url = str(config.get("openchronicle_mcp_url") or "").strip()
    return [
        {"name": "manual", "status": "ready"},
        {
            "name": "poll",
            "status": "configured" if configured == "poll" and poll_ready else "needs poll source",
        },
        {
            "name": "open_chronicle",
            "status": "configured" if configured == "open_chronicle" and open_chronicle_url else "needs MCP endpoint",
        },
    ]


def _split_command(line: str) -> tuple[str, str]:
    parts = line.split(maxsplit=1)
    command = parts[0].strip().lower()
    value = parts[1].strip() if len(parts) > 1 else ""
    return command, value


def _split_first(value: str) -> tuple[str, str]:
    parts = value.split(maxsplit=1)
    if not parts:
        return "", ""
    return parts[0], parts[1] if len(parts) > 1 else ""


def _parse_perceive_args(value: str) -> dict[str, Any]:
    tokens = shlex.split(value)
    options: dict[str, Any] = {
        "provider": None,
        "poll_url": None,
        "poll_command": None,
        "openchronicle_mcp_url": None,
        "openchronicle_since_minutes": None,
        "openchronicle_context_limit": None,
        "allow_command_poll": None,
        "decide_on_change": None,
        "timeout_seconds": None,
    }
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token == "--provider":
            options["provider"] = _next_value(tokens, index, token)
            index += 2
            continue
        if token == "--poll-url":
            options["poll_url"] = _next_value(tokens, index, token)
            index += 2
            continue
        if token == "--poll-command":
            options["poll_command"] = _next_value(tokens, index, token)
            index += 2
            continue
        if token == "--allow-command-poll":
            options["allow_command_poll"] = True
            index += 1
            continue
        if token == "--decide-on-change":
            options["decide_on_change"] = True
            index += 1
            continue
        if token == "--openchronicle-mcp-url":
            options["openchronicle_mcp_url"] = _next_value(tokens, index, token)
            index += 2
            continue
        if token == "--openchronicle-since-minutes":
            options["openchronicle_since_minutes"] = _positive_int(
                _next_value(tokens, index, token),
                token,
            )
            index += 2
            continue
        if token == "--openchronicle-context-limit":
            options["openchronicle_context_limit"] = _positive_int(
                _next_value(tokens, index, token),
                token,
            )
            index += 2
            continue
        if token == "--timeout":
            options["timeout_seconds"] = _positive_int(_next_value(tokens, index, token), token)
            index += 2
            continue
        raise ValueError(f"unknown /perceive option: {token}")
    return options


def _next_value(tokens: list[str], index: int, option: str) -> str:
    next_index = index + 1
    if next_index >= len(tokens) or tokens[next_index].startswith("--"):
        raise ValueError(f"{option} requires a value")
    return tokens[next_index]


def _positive_int(value: str, option: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{option} must be an integer") from exc
    if parsed <= 0:
        raise ValueError(f"{option} must be positive")
    return parsed
