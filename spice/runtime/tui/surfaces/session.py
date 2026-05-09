from __future__ import annotations

from typing import Any

from spice.runtime.session import SessionRecord, render_session_resume, render_session_stats, render_session_timeline
from spice.runtime.tui.theme import SpiceTheme


def render_session_panel(session: SessionRecord) -> Any:
    try:
        from rich import box
        from rich.panel import Panel
    except ImportError:
        return render_session_resume(session)
    return Panel(
        render_session_resume(session),
        title=f"[bold red]SESSION: {session.session_id}[/bold red]",
        border_style=SpiceTheme.PANEL_BORDER,
        box=box.ROUNDED,
    )


def render_timeline_panel(entries: list[Any], *, session_id: str) -> Any:
    try:
        from rich import box
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        return render_session_timeline(entries)

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("time", style=SpiceTheme.DIM)
    table.add_column("decision")
    table.add_column("status")
    for entry in entries:
        table.add_row(
            (entry.timestamp or "unknown")[:16],
            (entry.intent or entry.decision_id or entry.run_id or "unknown")[:48],
            Text(_status_chain(entry), style=_status_style(entry)),
        )
    return Panel(
        table if entries else Text("no runs found", style=SpiceTheme.DIM),
        title=f"[bold red]TIMELINE: {session_id}[/bold red]",
        border_style=SpiceTheme.PANEL_BORDER,
        box=box.ROUNDED,
    )


def render_stats_panel(stats: dict[str, Any]) -> Any:
    try:
        from rich import box
        from rich.panel import Panel
    except ImportError:
        return render_session_stats(stats)
    return Panel(
        render_session_stats(stats),
        title="[bold red]SESSION STATS[/bold red]",
        border_style=SpiceTheme.PANEL_BORDER,
        box=box.ROUNDED,
    )


def _status_chain(entry: Any) -> str:
    parts = []
    if entry.approval_status:
        parts.append(entry.approval_status)
    if entry.execution_status:
        parts.append(entry.execution_status)
    if entry.task_status:
        parts.append(entry.task_status)
    return " -> ".join(parts) if parts else "no status"


def _status_style(entry: Any) -> str:
    if entry.task_status == "success" or entry.execution_status in {"completed", "sdep_response_received"}:
        return SpiceTheme.SUCCESS
    if entry.approval_status == "rejected":
        return SpiceTheme.ERROR
    if entry.approval_status == "pending":
        return SpiceTheme.WARNING
    return SpiceTheme.DIM
