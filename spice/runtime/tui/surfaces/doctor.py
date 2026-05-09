from __future__ import annotations

from typing import Any

from spice.runtime.doctor import DoctorReport, render_doctor_report
from spice.runtime.tui.theme import SpiceTheme


def render_doctor_panel(report: DoctorReport) -> Any:
    try:
        from rich import box
        from rich.panel import Panel
        from rich.table import Table
    except ImportError:
        return render_doctor_report(report)

    table = Table(show_header=True, box=None, padding=(0, 1))
    table.add_column("check", style=SpiceTheme.DIM)
    table.add_column("status")
    table.add_column("detail")
    for check in report.checks:
        style = (
            SpiceTheme.SUCCESS
            if check.status == "ok"
            else SpiceTheme.WARNING
            if check.status == "warn"
            else SpiceTheme.ERROR
        )
        detail = check.detail
        if check.next_step:
            detail = f"{detail}\nNext: {check.next_step}"
        table.add_row(check.name, f"[{style}]{check.status}[/{style}]", detail)
    border = SpiceTheme.SUCCESS if report.status == "ok" else SpiceTheme.WARNING if report.status == "warn" else SpiceTheme.ERROR
    return Panel(
        table,
        title="[bold red]SPICE DOCTOR[/bold red]",
        subtitle=f"status: {report.status}",
        border_style=border,
        box=box.ROUNDED,
        padding=(1, 2),
    )
