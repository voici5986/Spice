from __future__ import annotations

import shutil
from io import StringIO
from typing import Any, Mapping

from spice.decision.compare import analyze_compare_payload, render_compare_text


def render_compare_rich(
    payload: Mapping[str, Any],
    *,
    show_execution: bool = False,
    use_bars: bool = True,
    force_terminal: bool | None = None,
    max_candidates: int = 3,
    max_why_not: int = 3,
    width: int | None = None,
) -> str:
    """Render a richer Decision Card, falling back silently when Rich is absent."""
    try:
        from rich import box
        from rich.console import Console, Group
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        return render_compare_text(
            payload,
            show_execution=show_execution,
            use_bars=use_bars,
        )

    analysis = analyze_compare_payload(payload)
    output = StringIO()
    render_width = _render_width(width)
    console = Console(
        file=output,
        force_terminal=force_terminal,
        color_system="standard" if force_terminal else None,
        width=render_width,
        legacy_windows=False,
    )

    header = _header_text(Text, analysis)
    body_items: list[Any] = [
        *(
            [_warnings_panel(Panel, Text, Group, box, analysis)]
            if analysis.get("warnings")
            else []
        ),
        _state_panel(Panel, Text, box, analysis),
        _candidates_panel(
            Panel,
            Text,
            Group,
            box,
            analysis,
            use_bars=use_bars,
            max_candidates=max_candidates,
        ),
        _selected_panel(Panel, Text, Group, box, analysis),
        _why_not_panel(Panel, Text, Group, box, analysis, max_why_not=max_why_not),
    ]
    if show_execution:
        body_items.append(_execution_panel(Panel, Text, Group, box, analysis))

    console.print(
        Panel(
            Group(header, *body_items),
            title="[bold red]SPICE DECISION CARD[/bold red]",
            border_style="red",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )
    return output.getvalue().rstrip()


def _warnings_panel(Panel: Any, Text: Any, Group: Any, box: Any, analysis: Mapping[str, Any]) -> Any:
    rendered = []
    for warning in list(analysis.get("warnings") or [])[:3]:
        text = Text()
        text.append(str(warning.get("message") or ""), style="yellow")
        if warning.get("reason"):
            text.append("\nReason: ", style="dim")
            text.append(str(warning.get("reason") or ""))
        rendered.append(text)
    return Panel(
        Group(*rendered),
        title="[bold yellow]WARNINGS[/bold yellow]",
        border_style="yellow",
        box=box.ROUNDED,
    )


def _render_width(width: int | None) -> int:
    if width is None:
        width = shutil.get_terminal_size((100, 24)).columns
    return max(32, min(int(width), 120))


def _header_text(Text: Any, analysis: Mapping[str, Any]) -> Any:
    text = Text()
    text.append("decision_id: ", style="dim")
    text.append(str(analysis["decision_id"]), style="white")
    text.append("\ntrace_ref: ", style="dim")
    text.append(str(analysis["trace_ref"]), style="white")
    return text


def _state_panel(Panel: Any, Text: Any, box: Any, analysis: Mapping[str, Any]) -> Any:
    summary = analysis["decision_relevant_state_summary"]
    text = Text()
    _append_line(text, "now", summary.get("now") or "unknown")
    if summary.get("available_window_minutes") is not None:
        _append_line(text, "window", f"{int(summary.get('available_window_minutes') or 0)} minutes")
    commitments = summary.get("active_commitments") or []
    work_items = summary.get("open_work_items") or []
    conflicts = summary.get("active_conflicts") or []
    _append_line(text, "commitments", str(len(commitments)))
    _append_line(text, "work_items", str(len(work_items)))
    _append_line(text, "conflicts", str(len(conflicts)))
    _append_line(text, "executor_available", str(bool(summary.get("executor_available"))).lower())
    return Panel(
        text,
        title="[bold]GENERAL STATE[/bold]",
        border_style="dim",
        box=box.ROUNDED,
    )


def _candidates_panel(
    Panel: Any,
    Text: Any,
    Group: Any,
    box: Any,
    analysis: Mapping[str, Any],
    *,
    use_bars: bool,
    max_candidates: int,
) -> Any:
    panels = []
    candidates = _visible_candidates(
        list(analysis["candidates"]),
        max_candidates=max_candidates,
    )
    for index, candidate in enumerate(candidates):
        prefix = chr(ord("A") + index)
        score = float(candidate["score_total"])
        title = f"{prefix}. {candidate['title']}"
        if candidate["is_selected"]:
            title = f"✓ {title}"
        if candidate["is_vetoed"]:
            title = f"blocked · {title}"
        lines = Text()
        recommendation = str(candidate.get("recommended_action") or candidate.get("intent") or "")
        if recommendation:
            lines.append("recommendation: ", style="dim")
            lines.append(recommendation)
        why_now = candidate.get("why_now") or []
        if why_now:
            lines.append("\nwhy now: ", style="dim")
            lines.append(_short_text_list(why_now))
        if candidate.get("enabled_reason"):
            lines.append("\navailable: ", style="dim")
            lines.append(str(candidate["enabled_reason"]))
        if candidate.get("expected_result"):
            lines.append("\nexpected outcome if chosen: ", style="dim")
            lines.append(str(candidate["expected_result"]))
        if candidate.get("executor_task"):
            lines.append("\nexecutor task: ", style="dim")
            lines.append(str(candidate["executor_task"]))
        affordance = candidate.get("execution_affordance") or {}
        if affordance:
            lines.append("\nexecution: ", style="dim")
            style = _execution_affordance_style(affordance)
            lines.append(_execution_affordance_summary(affordance), style=style)
        skill_resolution = candidate.get("skill_resolution") or {}
        if skill_resolution:
            lines.append("\nskill: ", style="dim")
            lines.append(_skill_resolution_summary(skill_resolution), style="dim")
        lines.append("\ninternal action: ", style="dim")
        lines.append(str(candidate["action"]), style="dim")
        lines.append("\nscore: ", style="dim")
        if use_bars:
            lines.append(_score_bar(Text, score))
            lines.append(f" {score:.2f}")
        else:
            lines.append(f"{score:.2f}")
        if candidate["is_vetoed"]:
            lines.append("\nveto: ", style="bold red")
            lines.append(_short_join(candidate["vetoes"], key="constraint_id"), style="red")
        else:
            lines.append("\nconstraints: ", style="dim")
            lines.append(_constraint_summary(candidate["constraints"]))
        effect = candidate.get("expected_effect") or {}
        lines.append("\nrisk: ", style="dim")
        lines.append(str(effect.get("commitment_risk") or "unknown"))
        simulation = candidate.get("simulation") or {}
        if simulation:
            lines.append("\nsimulation: ", style="dim")
            lines.append(str(simulation.get("expected_outcome") or simulation.get("simulated_outcome") or "recorded"))
            confidence = simulation.get("confidence")
            if confidence is not None:
                try:
                    lines.append(f" ({float(confidence):.2f})", style="dim")
                except (TypeError, ValueError):
                    pass
            if simulation.get("downside"):
                lines.append("\ndownside: ", style="dim")
                lines.append(str(simulation.get("downside")))
            if simulation.get("success_signal"):
                lines.append("\nsuccess: ", style="dim")
                lines.append(str(simulation.get("success_signal")))
            if simulation.get("time_fit"):
                lines.append("\ntime fit: ", style="dim")
                lines.append(str(simulation.get("time_fit")))
        history = candidate.get("history") or {}
        if int(history.get("similar_outcome_count") or 0) > 0:
            lines.append("\nhistory: ", style="dim")
            lines.append(_history_summary(history))
        border = "green" if candidate["is_selected"] else "red" if candidate["is_vetoed"] else "dim"
        panels.append(
            Panel(
                lines,
                title=title,
                border_style=border,
                box=box.ROUNDED,
            )
        )
    hidden_count = max(0, len(analysis["candidates"]) - len(candidates))
    if hidden_count:
        note = Text()
        note.append(f"+{hidden_count} lower-priority candidates hidden. ", style="dim")
        note.append("Use details/JSON output to inspect the full candidate set.", style="dim")
        panels.append(note)
    return Panel(
        Group(*panels),
        title="[bold]CANDIDATE DECISIONS[/bold]",
        border_style="red",
        box=box.ROUNDED,
    )


def _selected_panel(Panel: Any, Text: Any, Group: Any, box: Any, analysis: Mapping[str, Any]) -> Any:
    selected = analysis["selected_recommendation"]
    selected_text = Text()
    selected_text.append(str(selected["title"]), style="bold green")
    selected_text.append(f"\n{selected['candidate_id']}", style="dim")
    if selected.get("selection_reason"):
        selected_text.append("\n\nselection: ", style="dim")
        selected_text.append(str(selected["selection_reason"]))
    if selected.get("human_summary"):
        selected_text.append("\nrecommendation: ", style="dim")
        selected_text.append(str(selected["human_summary"]))

    basis_text = Text()
    basis_items = selected.get("decision_basis") or []
    if basis_items:
        for item in basis_items[:3]:
            basis_text.append("→ ", style="red")
            basis_text.append(_basis_summary(item))
            basis_text.append("\n")
    else:
        basis_text.append("→ No explicit selection basis was recorded.")
    return Panel(
        Group(
            Panel(selected_text, title="[bold green]SELECTED DECISION[/bold green]", border_style="green"),
            Panel(basis_text, title="[bold]WHY THIS WON[/bold]", border_style="green"),
        ),
        border_style="green",
        box=box.ROUNDED,
    )


def _why_not_panel(
    Panel: Any,
    Text: Any,
    Group: Any,
    box: Any,
    analysis: Mapping[str, Any],
    *,
    max_why_not: int,
) -> Any:
    rendered = []
    why_not = list(analysis["why_not_the_others"])
    for item in why_not[: max(0, max_why_not)]:
        text = Text()
        reasons = item.get("reasons") or []
        if reasons:
            for reason in reasons[:3]:
                style = "red" if str(reason.get("kind") or "") == "veto" else "yellow"
                text.append("• ", style=style)
                text.append(_why_not_summary(reason))
                text.append("\n")
        else:
            text.append("• No explicit compare evidence was recorded for this candidate.")
        rendered.append(
            Panel(
                text,
                title=f"{item['title']} ({item['candidate_id']})",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )
    hidden_count = max(0, len(why_not) - max(0, max_why_not))
    if hidden_count:
        note = Text()
        note.append(f"+{hidden_count} lower-priority alternatives hidden.", style="dim")
        rendered.append(note)
    if not rendered:
        rendered.append(Text("No non-selected candidates were recorded."))
    return Panel(
        Group(*rendered),
        title="[bold]WHY NOT OTHERS[/bold]",
        border_style="yellow",
        box=box.ROUNDED,
    )


def _visible_candidates(candidates: list[Mapping[str, Any]], *, max_candidates: int) -> list[Mapping[str, Any]]:
    if max_candidates <= 0 or len(candidates) <= max_candidates:
        return candidates
    selected = [candidate for candidate in candidates if candidate.get("is_selected")]
    non_selected = [candidate for candidate in candidates if not candidate.get("is_selected")]
    visible: list[Mapping[str, Any]] = []
    if selected:
        visible.append(selected[0])
    for candidate in non_selected:
        if len(visible) >= max_candidates:
            break
        visible.append(candidate)
    return visible


def _execution_panel(Panel: Any, Text: Any, Group: Any, box: Any, analysis: Mapping[str, Any]) -> Any:
    boundary = analysis["execution_boundary"]
    text = Text()
    _append_line(text, "requires_confirmation", str(boundary["requires_confirmation"]).lower())
    _append_line(text, "execution_path", boundary["execution_path"])
    if boundary.get("executor"):
        _append_line(text, "executor", boundary["executor"])
    if boundary.get("note"):
        _append_line(text, "note", boundary["note"])
    return Panel(
        text,
        title="[bold]EXECUTION BOUNDARY[/bold]",
        border_style="dim",
        box=box.ROUNDED,
    )


def _append_line(text: Any, label: str, value: Any) -> None:
    if len(text):
        text.append("\n")
    text.append(f"{label}: ", style="dim")
    text.append(str(value))


def _score_bar(Text: Any, score: float) -> Any:
    bounded = max(0.0, min(1.0, score))
    filled = int(round(bounded * 10))
    color = "green" if bounded >= 0.6 else "yellow" if bounded >= 0.3 else "red"
    text = Text()
    text.append("█" * filled, style=color)
    text.append("░" * (10 - filled), style="dim")
    return text


def _constraint_summary(constraints: list[Mapping[str, Any]]) -> str:
    if not constraints:
        return "none recorded"
    failed = [item for item in constraints if item.get("status") == "fail"]
    if failed:
        return _short_join(failed, key="constraint_id")
    return "all recorded hard constraints passed"


def _short_join(items: list[Mapping[str, Any]], *, key: str) -> str:
    values = [str(item.get(key) or "unknown") for item in items]
    return ", ".join(values[:3]) if values else "none"


def _short_text_list(items: list[Any]) -> str:
    values = [str(item) for item in items if str(item).strip()]
    return "; ".join(values[:2]) if values else "none"


def _history_summary(history: Mapping[str, Any]) -> str:
    count = int(history.get("similar_outcome_count") or 0)
    success = int(history.get("success_count") or 0)
    failed = int(history.get("failure_count") or 0)
    partial = int(history.get("partial_count") or 0)
    score = float(history.get("historical_score") or 0.0)
    parts = [f"{success}/{count} success"]
    if partial:
        parts.append(f"{partial} partial")
    if failed:
        parts.append(f"{failed} failed")
    parts.append(f"{score:.2f}")
    return ", ".join(parts)


def _execution_affordance_summary(affordance: Mapping[str, Any]) -> str:
    executor = affordance.get("executor") or {}
    permission = affordance.get("permission") or {}
    approval = affordance.get("approval") or {}
    executor_id = executor.get("executor_id") or "unknown"
    configured = permission.get("configured") or "unknown"
    required = permission.get("required") or "unknown"
    if not affordance.get("candidate_executable"):
        reason = str(affordance.get("blocked_reason") or "")
        if "execution_intent" in reason or "advisory" in reason.lower():
            return (
                "not executable; advisory only; no executor handoff requested; "
                "approval not required"
                f"{_capability_detail_suffix(affordance, advisory=True)}"
            )
        return (
            f"not executable; no executor handoff available: {reason or 'not approval eligible'}; "
            f"executor {executor_id}; permission {configured}->{required}; approval not available"
            f"{_capability_detail_suffix(affordance, missing=True)}"
        )
    approval_text = "approval required" if approval.get("required") else "approval not required"
    if affordance.get("blocked"):
        reason = affordance.get("blocked_reason") or "blocked"
        return (
            f"not executable yet; executor handoff blocked: {reason}; executor {executor_id}; "
            f"permission {configured}->{required}; {approval_text}"
            f"{_capability_detail_suffix(affordance, missing=True)}"
        )
    return (
        f"ready for approval via {executor_id}; "
        f"permission {configured}->{required}; {approval_text}"
        f"{_capability_detail_suffix(affordance)}"
    )


def _capability_detail_suffix(
    affordance: Mapping[str, Any],
    *,
    advisory: bool = False,
    missing: bool = False,
) -> str:
    capability = affordance.get("capability") or {}
    if not isinstance(capability, Mapping):
        capability = {}
    required = str(
        capability.get("required_capability")
        or affordance.get("required_capability")
        or ""
    ).strip()
    source = str(
        capability.get("source")
        or affordance.get("executor_capability_source")
        or ""
    ).strip()
    matched = str(capability.get("matched_capability") or "").strip()
    limitations = [
        str(item).strip()
        for item in (capability.get("limitations") or [])
        if str(item).strip()
    ]
    parts: list[str] = []
    if required and not advisory:
        if missing or not capability.get("executor_has_required_capability"):
            parts.append(f"missing capability {required}")
        else:
            matched_text = f" matched {matched}" if matched else ""
            parts.append(f"required capability {required}{matched_text}")
    if source:
        parts.append(f"capability source {source}")
    if limitations:
        parts.append("limitations " + "; ".join(limitations[:2]))
    return "; " + "; ".join(parts) if parts else ""


def _execution_affordance_style(affordance: Mapping[str, Any]) -> str:
    if not affordance.get("candidate_executable"):
        reason = str(affordance.get("blocked_reason") or "")
        if "execution_intent" in reason or "advisory" in reason.lower():
            return "dim"
        return "yellow"
    return "red" if affordance.get("blocked") else "green"


def _skill_resolution_summary(skill_resolution: Mapping[str, Any]) -> str:
    status = str(skill_resolution.get("status") or "unknown")
    resolved = skill_resolution.get("resolved_skill")
    if isinstance(resolved, Mapping):
        skill_id = str(resolved.get("skill_id") or "unknown")
        executor_id = str(resolved.get("executor_id") or "unknown")
        metadata = resolved.get("metadata")
        source = str(metadata.get("skill_source") or "") if isinstance(metadata, Mapping) else ""
        suffix = f"; source {source}" if source else ""
        return f"{skill_id} via {executor_id}{suffix}"
    reasons = skill_resolution.get("unresolved_reasons")
    if isinstance(reasons, list) and reasons:
        return f"{status}: {str(reasons[0])}"
    return status


def _basis_summary(reason: Mapping[str, Any]) -> str:
    if reason.get("summary"):
        return str(reason["summary"])
    kind = str(reason.get("kind") or "")
    if kind == "weighted_dimension":
        return (
            f"{reason.get('label')} carried weight "
            f"{float(reason.get('weight', 0.0)):.2f} "
            f"with contribution {float(reason.get('contribution', 0.0)):.2f}."
        )
    if kind == "tradeoff_rule":
        return f"tradeoff rule {reason.get('rule_id')} supported this candidate."
    return "additional trace evidence supported this candidate."


def _why_not_summary(reason: Mapping[str, Any]) -> str:
    if reason.get("summary"):
        return str(reason["summary"])
    kind = str(reason.get("kind") or "")
    if kind == "veto":
        return f"vetoed by {reason.get('constraint_id')}: {reason.get('reason')}"
    if kind == "tradeoff_rule":
        return f"tradeoff rule {reason.get('rule_id')}: {reason.get('reason')}"
    if kind == "weighted_dimension_gap":
        return (
            f"lower {reason.get('label')} contribution "
            f"({float(reason.get('candidate_contribution', 0.0)):.2f} vs "
            f"{float(reason.get('selected_contribution', 0.0)):.2f})."
        )
    return "no explicit why-not reason was recorded."
