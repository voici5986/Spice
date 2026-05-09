from __future__ import annotations

from typing import Any

from spice.runtime.tui.theme import SpiceTheme


def render_execution_panel(artifact: dict[str, Any], rendered_text: str) -> Any:
    try:
        from rich import box
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        return rendered_text

    provider = str(artifact.get("executor_provider") or "executor")
    title = "DRY-RUN EXECUTION" if provider == "dry_run" else "EXECUTION RESULT"
    status_style = _status_style(str(artifact.get("task_status") or artifact.get("protocol_status") or ""))

    ids = Table(show_header=False, box=None, padding=(0, 1))
    ids.add_column("key", style=SpiceTheme.DIM)
    ids.add_column("value")
    for key in (
        "approval_id",
        "decision_id",
        "trace_ref",
        "candidate_id",
        "execution_id",
        "request_id",
        "outcome_id",
    ):
        value = artifact.get(key)
        if value:
            ids.add_row(f"{key}:", str(value))

    boundary = Table(show_header=False, box=None, padding=(0, 1))
    boundary.add_column("key", style=SpiceTheme.DIM)
    boundary.add_column("value")
    boundary.add_row("executor_provider:", provider)
    if artifact.get("executor_command"):
        boundary.add_row("command:", str(artifact.get("executor_command")))
    boundary.add_row("planned_executor:", str(artifact.get("executor_id") or "unknown"))
    boundary.add_row("skill:", str(artifact.get("skill_id") or "unknown"))
    boundary.add_row("context_pack_id:", str(artifact.get("context_pack_id") or "unknown"))
    boundary.add_row("sdep_request_sent:", _bool_text(Text, artifact.get("sdep_request_sent")))
    boundary.add_row("executor_called:", _bool_text(Text, artifact.get("executor_called")))
    boundary.add_row("real_executor_called:", _bool_text(Text, artifact.get("real_executor_called")))
    boundary.add_row("executed:", _bool_text(Text, artifact.get("executed")))

    outcome = Table(show_header=False, box=None, padding=(0, 1))
    outcome.add_column("key", style=SpiceTheme.DIM)
    outcome.add_column("value")
    outcome.add_row("protocol_status:", Text(str(artifact.get("protocol_status") or "unknown"), style=status_style))
    outcome.add_row("task_status:", Text(str(artifact.get("task_status") or "unknown"), style=status_style))
    outcome.add_row("state_updated:", _bool_text(Text, artifact.get("state_updated")))
    outcome.add_row("persisted:", _bool_text(Text, artifact.get("persisted")))
    if artifact.get("state_after_ref"):
        outcome.add_row("state_after_ref:", str(artifact.get("state_after_ref")))

    body = Group(
        Text("approved decision -> execution boundary -> outcome -> state feedback", style=SpiceTheme.DIM),
        ids,
        Panel(boundary, title="[bold red]EXECUTION BOUNDARY[/bold red]", border_style=SpiceTheme.PANEL_BORDER, box=box.ROUNDED),
        Panel(outcome, title="[bold red]OUTCOME[/bold red]", border_style=SpiceTheme.PANEL_BORDER, box=box.ROUNDED),
    )
    return Panel(
        body,
        title=f"[bold red]{title}[/bold red]",
        border_style=SpiceTheme.PANEL_BORDER,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def render_execution_dispatch_panel(
    *,
    approval_id: str,
    executor_provider: str,
    executor_command: str,
) -> Any:
    text = (
        "Spice is crossing the execution boundary.\n\n"
        f"approval_id: {approval_id}\n"
        f"executor: {executor_provider}\n"
        f"command: {executor_command or 'configured runtime'}\n\n"
        "building SDEP execute.request -> sending to executor -> waiting for execute.response"
    )
    try:
        from rich import box
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        return f"DISPATCHING TO EXECUTOR\n{text}"
    return Panel(
        Text(text),
        title="[bold red]DISPATCHING TO EXECUTOR[/bold red]",
        border_style=SpiceTheme.PANEL_BORDER,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def render_execution_summary_panel(artifact: dict[str, Any]) -> Any:
    provider = str(artifact.get("executor_provider") or "executor")
    task_status = str(artifact.get("task_status") or "unknown")
    protocol_status = str(artifact.get("protocol_status") or "unknown")
    state_updated = bool(artifact.get("state_updated"))
    persisted = bool(artifact.get("persisted"))
    outcome_id = str(artifact.get("outcome_id") or "")

    if task_status.lower() in {"success", "ok", "completed"}:
        title = "EXECUTION COMPLETE"
        body = [
            f"{provider} finished successfully.",
            "",
            "Spice received a valid SDEP execute.response, recorded the outcome, "
            "updated state, and attached the execution result to this session.",
        ]
        border = SpiceTheme.SUCCESS
    elif protocol_status.lower() == "success":
        title = "EXECUTION RETURNED FAILURE"
        body = [
            f"{provider} returned a failed task outcome.",
            "",
            "Spice received a valid SDEP execute.response and recorded the failure. "
            "The result can influence future decisions.",
        ]
        border = SpiceTheme.ERROR
    else:
        title = "EXECUTION STATUS UNKNOWN"
        body = [
            f"{provider} returned protocol status {protocol_status}.",
            "",
            "Spice recorded the executor response, but the outcome should be inspected.",
        ]
        border = SpiceTheme.WARNING

    body.extend(
        [
            "",
            f"task_status: {task_status}",
            f"state_updated: {str(state_updated).lower()}",
            f"persisted: {str(persisted).lower()}",
        ]
    )
    if outcome_id:
        body.append(f"outcome_id: {outcome_id}")
    body.extend(["", "Next: /session to review, /state to inspect updated state, or type a new intent."])
    return _summary_panel(title=title, body="\n".join(body), border_style=border)


def render_execution_error_panel(
    *,
    approval_id: str,
    executor_provider: str,
    error: Exception | str,
) -> Any:
    message = str(error)
    lowered = message.lower()
    if "timed out" in lowered or "timeout" in lowered:
        headline = f"{executor_provider} timed out before returning a valid SDEP response."
        next_step = f"Next: retry with /execute {approval_id}, check network/VPN, or switch executor."
    else:
        headline = f"{executor_provider} did not complete execution."
        next_step = f"Next: inspect with /details {approval_id}, fix the executor, then retry /execute {approval_id}."
    body = "\n".join(
        [
            headline,
            "",
            "The approval remains approved, but Spice did not record a successful executor outcome.",
            "",
            f"approval_id: {approval_id}",
            f"error: {message}",
            "",
            next_step,
        ]
    )
    return _summary_panel(title="EXECUTION DID NOT COMPLETE", body=body, border_style=SpiceTheme.ERROR)


def _summary_panel(*, title: str, body: str, border_style: str) -> Any:
    try:
        from rich import box
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        return f"{title}\n{body}"
    return Panel(
        Text(body),
        title=f"[bold red]{title}[/bold red]",
        border_style=border_style,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _bool_text(Text: Any, value: Any) -> Any:
    enabled = bool(value)
    return Text(str(enabled).lower(), style=SpiceTheme.SUCCESS if enabled else SpiceTheme.DIM)


def _status_style(status: str) -> str:
    lowered = status.lower()
    if lowered in {"success", "ok", "completed"}:
        return SpiceTheme.SUCCESS
    if lowered in {"failed", "error"}:
        return SpiceTheme.ERROR
    return SpiceTheme.WARNING
