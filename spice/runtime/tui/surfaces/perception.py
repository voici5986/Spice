from __future__ import annotations

from typing import Any

from spice.runtime.perceive import render_perceive_text
from spice.runtime.tui.theme import SpiceTheme


def render_perception_panel(artifact: dict[str, Any]) -> Any:
    try:
        from rich import box
        from rich.console import Group
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        return render_perceive_text(artifact)

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("key", style=SpiceTheme.DIM)
    table.add_column("value")
    table.add_row("provider:", str(artifact.get("provider") or "unknown"))
    table.add_row("observations:", str(artifact.get("observation_count") or 0))
    table.add_row("changed:", str(artifact.get("changed_count") or 0))
    table.add_row("deduped:", str(artifact.get("deduped_count") or 0))
    table.add_row("decision_triggered:", _bool_text(Text, artifact.get("decision_triggered")))
    if artifact.get("run_id"):
        table.add_row("run_id:", str(artifact.get("run_id")))
    if artifact.get("decision_id"):
        table.add_row("decision_id:", str(artifact.get("decision_id")))
    if artifact.get("approval_id"):
        table.add_row("approval_id:", str(artifact.get("approval_id")))
    table.add_row("executor_called:", _bool_text(Text, artifact.get("executor_called")))
    table.add_row("sdep_request_sent:", _bool_text(Text, artifact.get("sdep_request_sent")))

    observations = artifact.get("observations")
    observation_text = Text()
    if isinstance(observations, list) and observations:
        for observation in observations[:5]:
            if not isinstance(observation, dict):
                continue
            observation_text.append("• ", style=SpiceTheme.DIM)
            observation_text.append(str(observation.get("summary") or observation.get("observation_id") or "signal"))
            observation_text.append("\n")
    else:
        observation_text.append("no changed observations", style=SpiceTheme.DIM)

    next_text = Text()
    if artifact.get("decision_triggered"):
        next_text.append("Next: ", style=SpiceTheme.DIM)
        next_text.append("/approvals", style=SpiceTheme.SUCCESS)
        if artifact.get("approval_id"):
            next_text.append(f" or /approve {artifact.get('approval_id')}", style=SpiceTheme.SUCCESS)
    else:
        next_text.append("State updated only. Use /perceive --decide-on-change to open a Decision Card on changes.", style=SpiceTheme.DIM)

    body = Group(
        table,
        Panel(
            observation_text,
            title="[bold red]SIGNALS[/bold red]",
            border_style=SpiceTheme.PANEL_BORDER,
            box=box.ROUNDED,
        ),
        next_text,
    )
    return Panel(
        body,
        title="[bold red]PERCEPTION[/bold red]",
        border_style=SpiceTheme.PANEL_BORDER,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _bool_text(Text: Any, value: Any) -> Any:
    enabled = bool(value)
    return Text(str(enabled).lower(), style=SpiceTheme.SUCCESS if enabled else SpiceTheme.DIM)
