from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any

from spice.runtime.tui.theme import SpiceTheme


@dataclass(frozen=True, slots=True)
class ProgressStep:
    key: str
    icon: str
    label: str
    detail: str = ""
    skipped: bool = False


class TUIProgressFlow:
    def __init__(
        self,
        *,
        console: Any,
        title: str,
        steps: list[ProgressStep],
        interval_seconds: float = 0.7,
    ) -> None:
        self.console = console
        self.title = title
        self.steps = steps
        self.interval_seconds = interval_seconds
        self._live: Any = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._failed = False
        self._active_index = 0

    def __enter__(self) -> "TUIProgressFlow":
        if self.console is None:
            return self
        try:
            from rich.live import Live
        except ImportError:
            return self
        self._live = Live(
            self._render(),
            console=self.console,
            refresh_per_second=8,
            transient=True,
        )
        self._live.start()
        self._thread = threading.Thread(target=self._advance_until_stopped, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self._failed = exc_type is not None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._live is not None:
            self._active_index = len(self.steps)
            self._live.update(self._render())
            self._live.stop()

    def _advance_until_stopped(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            if self._active_index < max(len(self.steps) - 1, 0):
                self._active_index += 1
            if self._live is not None:
                self._live.update(self._render())

    def _render(self) -> Any:
        try:
            from rich import box
            from rich.console import Group
            from rich.panel import Panel
            from rich.text import Text
        except ImportError:
            return self.title

        lines: list[Text] = []
        total = len(self.steps)
        for index, step in enumerate(self.steps):
            status_icon, style = self._status_for_step(index, step)
            detail = f"  {step.detail}" if step.detail else ""
            lines.append(
                Text(
                    f"{status_icon} {step.icon} {index + 1}/{total} {step.label}{detail}",
                    style=style,
                )
            )
        return Panel(
            Group(*lines),
            title=f"[bold red]{self.title}[/bold red]",
            border_style=SpiceTheme.PANEL_BORDER,
            box=box.ROUNDED,
            padding=(1, 2),
        )

    def _status_for_step(self, index: int, step: ProgressStep) -> tuple[str, str]:
        if self._failed:
            if index < self._active_index:
                return "✓", SpiceTheme.SUCCESS
            if index == self._active_index:
                return "x", SpiceTheme.ERROR
            return "·", SpiceTheme.DIM
        if step.skipped:
            return "!", SpiceTheme.WARNING
        if self._active_index >= len(self.steps) or index < self._active_index:
            return "✓", SpiceTheme.SUCCESS
        if index == self._active_index:
            return "⠋", SpiceTheme.WARNING
        return "·", SpiceTheme.DIM


def decision_progress(console: Any, *, config: dict[str, Any], mode: str) -> TUIProgressFlow:
    llm_provider = str(config.get("llm_provider") or "deterministic")
    llm_model = str(config.get("llm_model") or "").strip()
    llm_label = f"{llm_provider}/{llm_model}" if llm_model else llm_provider
    llm_expand = bool(config.get("llm_candidate_expand"))
    llm_simulation = bool(config.get("llm_simulation"))
    ready_detail = llm_label if llm_provider != "deterministic" else "deterministic runtime"
    selection_detail = (
        "approval-eligible execution candidate"
        if mode == "act"
        else "advisory recommendation"
        if mode == "advise"
        else "best available recommendation"
    )
    steps = [
        ProgressStep("intent", "?", "Understanding intent"),
        ProgressStep("state", "#", "Updating world state"),
        ProgressStep("candidates", "+", "Generating rule candidates"),
        ProgressStep(
            "llm_expand",
            "*",
            "Expanding candidates with LLM" if llm_expand else "Skipping LLM expansion",
            ready_detail if llm_expand else "disabled",
            skipped=not llm_expand,
        ),
        ProgressStep(
            "simulation",
            "~",
            "Simulating outcomes" if llm_simulation else "Skipping simulation",
            ready_detail if llm_simulation else "disabled",
            skipped=not llm_simulation,
        ),
        ProgressStep("score", "=", "Scoring tradeoffs and building Decision Card", selection_detail),
    ]
    return TUIProgressFlow(console=console, title="SPICE DECISION FLOW", steps=steps)


def execution_progress(
    console: Any,
    *,
    approval_id: str,
    executor: Any,
) -> TUIProgressFlow:
    command = str(getattr(executor, "command", "") or "configured runtime")
    permission = str(getattr(executor, "permission_mode", "") or "workspace_write")
    executor_id = str(getattr(executor, "executor_id", "") or "executor")
    steps = [
        ProgressStep("approval", "^", "Approval boundary passed", approval_id),
        ProgressStep("runtime", "@", "Runtime resolved", f"{executor_id}; permission={permission}"),
        ProgressStep("request", ">", "Building SDEP execute.request", command),
        ProgressStep("wait", ">", f"Waiting for {executor_id} execute.response"),
        ProgressStep("outcome", "<", "Recording outcome and updating state"),
    ]
    return TUIProgressFlow(console=console, title="SPICE EXECUTION FLOW", steps=steps)
