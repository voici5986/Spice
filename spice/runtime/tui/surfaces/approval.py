from __future__ import annotations

from typing import Any

from spice.decision.general.approval import Approval
from spice.runtime.approval_flow import (
    ApprovalResolutionResult,
    render_approval_details,
    render_approval_list,
    render_approval_resolution,
)
from spice.runtime.tui.theme import SpiceTheme


def render_approvals_panel(approvals: list[Approval]) -> Any:
    try:
        from rich import box
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        return render_approval_list(approvals)

    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("approval_id", style=SpiceTheme.DIM)
    table.add_column("status")
    table.add_column("candidate")
    counts: dict[str, int] = {}
    for approval in approvals:
        counts[approval.status] = counts.get(approval.status, 0) + 1
        table.add_row(
            approval.approval_id,
            _status_text(Text, approval.status),
            approval.candidate_id or "none",
        )
    footer = Text()
    footer.append(
        f"{counts.get('pending', 0)} pending · "
        f"{counts.get('approved', 0)} approved · "
        f"{counts.get('rejected', 0)} rejected",
        style=SpiceTheme.DIM,
    )
    return Panel(
        table if approvals else Text("no approvals found", style=SpiceTheme.DIM),
        title="[bold red]APPROVALS[/bold red]",
        subtitle=footer,
        border_style=SpiceTheme.PANEL_BORDER,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def render_approval_details_panel(approval: Approval) -> Any:
    try:
        from rich import box
        from rich.panel import Panel
    except ImportError:
        return render_approval_details(approval)
    return Panel(
        render_approval_details(approval),
        title="[bold red]APPROVAL[/bold red]",
        border_style=SpiceTheme.PANEL_BORDER,
        box=box.ROUNDED,
    )


def render_approval_resolution_panel(result: ApprovalResolutionResult) -> Any:
    try:
        from rich import box
        from rich.panel import Panel
    except ImportError:
        return render_approval_resolution(result)
    status = result.approval.status
    border = SpiceTheme.SUCCESS if status == "approved" else SpiceTheme.ERROR if status == "rejected" else SpiceTheme.WARNING
    return Panel(
        render_approval_resolution(result),
        title=f"[bold red]APPROVAL {status.upper()}[/bold red]",
        border_style=border,
        box=box.ROUNDED,
    )


def _status_text(Text: Any, status: str) -> Any:
    style = (
        SpiceTheme.SUCCESS
        if status == "approved"
        else SpiceTheme.ERROR
        if status == "rejected"
        else SpiceTheme.WARNING
        if status == "pending"
        else SpiceTheme.DIM
    )
    return Text(status, style=style)
