from __future__ import annotations

from typing import Any

from spice.runtime.tui.theme import SpiceTheme


def render_state_panel(state_payload: dict[str, Any]) -> Any:
    general = _general_state_payload(state_payload)
    counts = {
        "observations": len(_items(general, "observations")),
        "intents": len(_items(general, "intents")),
        "work_items": len(_items(general, "work_items")),
        "commitments": len(_items(general, "commitments")),
        "risks": len(_items(general, "risks")),
        "open_loops": len(_items(general, "open_loops")),
        "outcomes": len(_items(general, "outcomes")),
        "approvals": len(_items(general, "approvals")),
    }
    try:
        from rich import box
        from rich.console import Group
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        lines = ["WORLD STATE"]
        lines.extend(f"- {key}: {value}" for key, value in counts.items())
        return "\n".join(lines)

    body = Text()
    for key, value in counts.items():
        body.append(f"{key}: ", style=SpiceTheme.DIM)
        body.append(str(value))
        body.append("\n")
    details = Text()
    _append_samples(details, "Open work items", _items(general, "work_items"), "title")
    _append_samples(details, "Active commitments", _items(general, "commitments"), "title")
    _append_samples(details, "Open loops", _items(general, "open_loops"), "summary")
    return Panel(
        Group(body, details),
        title="[bold red]WORLD STATE[/bold red]",
        border_style=SpiceTheme.PANEL_BORDER,
        box=box.ROUNDED,
        padding=(1, 2),
    )


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


def _append_samples(text: Any, title: str, items: list[dict[str, Any]], field: str) -> None:
    if not items:
        return
    text.append(f"\n{title}:\n", style=SpiceTheme.DIM)
    for item in items[:3]:
        text.append("  • ", style=SpiceTheme.DIM)
        text.append(str(item.get(field) or item.get("status") or "unknown"))
        text.append("\n")
